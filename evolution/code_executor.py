"""
代码执行器 — 参考 AIDEML Interpreter
=====================================
独立进程执行 Python 代码，有超时保护、stdout/stderr 捕获、异常栈记录。

参考: https://github.com/WecoAI/aideml/blob/main/aide/interpreter.py
"""

import logging
import os
import queue
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

logger = logging.getLogger("evolution.interpreter")


@dataclass
class ExecutionResult:
    """代码执行结果"""
    term_out: list[str]       # stdout + stderr 输出
    exec_time: float          # 执行时间 (秒)
    exc_type: str | None      # 异常类型名
    exc_info: dict | None     # 异常信息
    exc_stack: list[tuple] | None  # 异常栈


def _exception_summary(e, working_dir, exec_file_name):
    """生成异常摘要"""
    tb_lines = traceback.format_exception(e)
    # 过滤掉框架内部的栈帧
    tb_str = "".join(
        line for line in tb_lines
        if "evolution/" not in line and "importlib" not in line
    )
    tb_str = tb_str.replace(str(working_dir / exec_file_name), exec_file_name)

    exc_info = {}
    if hasattr(e, "args"):
        exc_info["args"] = [str(i) for i in e.args]
    for att in ["name", "msg", "obj"]:
        if hasattr(e, att):
            exc_info[att] = str(getattr(e, att))

    tb = traceback.extract_tb(e.__traceback__)
    exc_stack = [(t.filename, t.lineno, t.name, t.line) for t in tb]

    return tb_str, e.__class__.__name__, exc_info, exc_stack


class _RedirectQueue:
    """将 stdout/stderr 重定向到队列"""
    def __init__(self, q, timeout=5):
        self.queue = q
        self.timeout = timeout

    def write(self, msg):
        try:
            self.queue.put(msg, timeout=self.timeout)
        except queue.Full:
            pass

    def flush(self):
        pass


class CodeInterpreter:
    """
    独立进程代码执行器。

    参考 AIDEML 的 Interpreter，在隔离子进程中执行 Python 代码：
    - 超时保护 (默认 60 秒)
    - 捕获 stdout/stderr
    - 记录异常类型和栈
    - 支持重置/复用会话
    """

    def __init__(self, working_dir: str | Path = None,
                 timeout: int = 60,
                 agent_file_name: str = "runfile.py"):
        self.working_dir = Path(working_dir or tempfile.gettempdir()).resolve()
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.agent_file_name = agent_file_name
        self.process: Process | None = None
        self.code_inq: Queue | None = None
        self.result_outq: Queue | None = None
        self.event_outq: Queue | None = None

    def _child_setup(self, result_outq: Queue):
        """子进程初始化"""
        os.chdir(str(self.working_dir))
        sys.path.append(str(self.working_dir))
        sys.stdout = sys.stderr = _RedirectQueue(result_outq)

    def _run_session(self, code_inq: Queue, result_outq: Queue, event_outq: Queue):
        """子进程主循环"""
        self._child_setup(result_outq)
        global_scope = {
            "__name__": "__main__",
            "__file__": self.agent_file_name,
            "__builtins__": __builtins__,
        }
        while True:
            code = code_inq.get()
            os.chdir(str(self.working_dir))

            # 写入文件
            with open(self.agent_file_name, "w") as f:
                f.write(code)

            event_outq.put(("state:ready",))
            try:
                exec(compile(code, self.agent_file_name, "exec"), global_scope)
            except BaseException as e:
                tb_str, e_cls, exc_info, exc_stack = _exception_summary(
                    e, self.working_dir, self.agent_file_name
                )
                result_outq.put(tb_str)
                if e_cls == "KeyboardInterrupt":
                    e_cls = "TimeoutError"
                event_outq.put(("state:finished", e_cls, exc_info, exc_stack))
            else:
                event_outq.put(("state:finished", None, None, None))

            # 清理文件
            try:
                os.remove(self.agent_file_name)
            except OSError:
                pass

            result_outq.put(None)  # EOF 标记

    def _create_process(self):
        """创建子进程"""
        self.code_inq = Queue()
        self.result_outq = Queue()
        self.event_outq = Queue()
        self.process = Process(
            target=self._run_session,
            args=(self.code_inq, self.result_outq, self.event_outq),
        )
        self.process.start()

    def cleanup(self):
        """清理子进程"""
        if self.process is None:
            return
        try:
            self.process.terminate()
            self.process.join(timeout=1)
            if self.process.exitcode is None:
                self.process.kill()
                self.process.join(timeout=1)
        except Exception as e:
            logger.warning(f"清理子进程失败: {e}")
        finally:
            if self.process is not None:
                self.process.close()
                self.process = None

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """
        在子进程中执行代码。

        Args:
            code: Python 代码
            reset_session: 是否重置会话

        Returns:
            ExecutionResult
        """
        if reset_session:
            if self.process is not None:
                self.cleanup()
            self._create_process()
        else:
            assert self.process is not None

        assert self.process.is_alive()

        self.code_inq.put(code)

        # 等待子进程开始执行
        try:
            state = self.event_outq.get(timeout=10)
        except queue.Empty:
            return ExecutionResult(
                term_out=["REPL 子进程启动超时"],
                exec_time=0,
                exc_type="TimeoutError",
                exc_info=None,
                exc_stack=None,
            )
        assert state[0] == "state:ready"

        start_time = time.time()
        timed_out = False

        # 收集输出
        term_out = []
        while True:
            try:
                msg = self.result_outq.get(timeout=self.timeout)
            except queue.Empty:
                timed_out = True
                break
            if msg is None:
                break
            term_out.append(msg)

        if timed_out:
            self.cleanup()
            return ExecutionResult(
                term_out=term_out + [f"\n[超时] 执行超过 {self.timeout} 秒"],
                exec_time=self.timeout,
                exc_type="TimeoutError",
                exc_info=None,
                exc_stack=None,
            )

        # 等待执行完成事件
        try:
            event = self.event_outq.get(timeout=5)
        except queue.Empty:
            event = ("state:finished", "UnknownError", None, None)

        exec_time = time.time() - start_time
        exc_type = event[1] if len(event) > 1 else None
        exc_info = event[2] if len(event) > 2 else None
        exc_stack = event[3] if len(event) > 3 else None

        return ExecutionResult(
            term_out=term_out,
            exec_time=exec_time,
            exc_type=exc_type,
            exc_info=exc_info,
            exc_stack=exc_stack,
        )

    def __del__(self):
        self.cleanup()


# 便捷函数
def execute_code(code: str, working_dir: str = None, timeout: int = 60) -> ExecutionResult:
    """执行代码并返回结果"""
    import tempfile
    if working_dir is None:
        working_dir = tempfile.mkdtemp(prefix="evo_")
    interpreter = CodeInterpreter(working_dir=working_dir, timeout=timeout)
    try:
        return interpreter.run(code)
    finally:
        interpreter.cleanup()


def is_valid_python(code: str) -> bool:
    """验证 Python 代码语法"""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False

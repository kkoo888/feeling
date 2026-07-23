class FusionEngineV2:
    def __init__(self):
        self.intent_patterns = {
            'translation': [
                r'翻译', r'translate', r'转换', r'convert',
                r'中文', r'英文', r'日文', r'语言'
            ],
            'debug': [
                r'调试', r'debug', r'错误', r'error', r'修复', r'fix',
                r'问题', r'bug', r'异常', r'exception'
            ],
            'deploy': [
                r'部署', r'deploy', r'发布', r'publish',
                r'上线', r'生产环境', r'production'
            ],
            'analyze': [
                r'分析', r'analyze', r'解析', r'parse',
                r'评估', r'evaluate', r'检测'
            ]
        }
        
        self.entity_patterns = {
            'file': [
                r'[\w-]+\.py', r'[\w-]+\.js', r'[\w-]+\.java',
                r'[\w-]+\.c\+\+', r'[\w-]+\.cpp', r'[\w-]+\.h',
                r'[\w-]+\.txt', r'[\w-]+\.json', r'[\w-]+\.yaml',
                r'[\w-]+\.yml', r'[\w-]+\.toml', r'[\w-]+\.cfg'
            ],
            'code': [
                r'函数\s*\(', r'function\s*\(', r'class\s*\w+',
                r'import\s+\w+', r'from\s+\w+\s+import'
            ]
        }
        
        self.fusion_weights = {
            'intent': 0.4,
            'entity': 0.3,
            'context': 0.3
        }
        
        self.chain_steps = ['分解问题', '信息提取', '推理计算', '结果生成']
        
    def identify_intent(self, text):
        intents = []
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text.lower():
                    intents.append(intent)
                    break
        
        if '翻译' in text and 'python' in text:
            intents.append('translation')
        if '调试' in text and ('代码' in text or '程序' in text):
            intents.append('debug')
        
        return intents if intents else ['general']
    
    def extract_entities(self, text):
        entities = {'files': [], 'code': [], 'others': []}
        
        import re
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                if entity_type == 'file':
                    entities['files'].extend(matches)
                elif entity_type == 'code':
                    entities['code'].extend(matches)
        
        return entities
    
    def filter_information(self, content, intents, entities):
        if not content:
            return content
        
        relevant = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if any(intent in line.lower() for intent in intents):
                relevant.append(line)
            elif any(entity in line for entity in entities['files']):
                relevant.append(line)
            elif len(line) > 20 and not line.startswith(('#', '//', '/*')):
                relevant.append(line)
        
        return '\n'.join(relevant[:10])
    
    def chain_reasoning(self, query, intents, entities):
        chain = []
        current_step = 0
        intermediate_results = {}
        
        while current_step < len(self.chain_steps):
            step_name = self.chain_steps[current_step]
            
            if step_name == '分解问题':
                subtasks = []
                for intent in intents:
                    subtasks.append(f"执行{intent}操作")
                if entities['files']:
                    subtasks.append(f"处理文件: {', '.join(entities['files'][:3])}")
                chain.append({'step': step_name, 'subtasks': subtasks})
                intermediate_results['subtasks'] = subtasks
                
            elif step_name == '信息提取':
                extracted = {
                    'intent_confidence': len(intents) * 0.2,
                    'entity_coverage': min(len(entities['files']), 5) / 5,
                    'query_complexity': min(len(query.split()), 10) / 10
                }
                chain.append({'step': step_name, 'results': extracted})
                intermediate_results['metrics'] = extracted
                
            elif step_name == '推理计算':
                scores = {}
                for intent in intents:
                    scores[intent] = 0.8 + (0.1 if len(entities['files']) > 0 else 0)
                
                total_score = sum(scores.values()) / max(len(scores), 1)
                weights = {
                    'intent': len(intents) * 0.2,
                    'entity': min(len(entities['files']), 3) * 0.3,
                    'coherence': total_score
                }
                chain.append({'step': step_name, 'scores': scores, 'weights': weights})
                intermediate_results['fusion_score'] = total_score
                
            elif step_name == '结果生成':
                final_score = intermediate_results.get('fusion_score', 0.5) * 100
                output_template = f"识别到{len(intents)}个意图，提取{len(entities['files'])}个文件实体"
                if final_score > 70:
                    output_template += "\n建议执行相关操作"
                chain.append({'step': step_name, 'output': output_template, 'confidence': final_score})
                
            current_step += 1
        
        return chain
    
    def fuse_information(self, intents, entities, context):
        intent_score = len(intents) * self.fusion_weights['intent']
        entity_score = min(len(entities['files']), 5) * self.fusion_weights['entity']
        context_score = len(context.split()) * self.fusion_weights['context'] * 0.1
        
        total_score = min(intent_score + entity_score + context_score, 1.0)
        
        fusion_result = {
            'intents': intents,
            'entities': entities,
            'score': total_score,
            'metadata': {
                'intent_weight': self.fusion_weights['intent'],
                'entity_weight': self.fusion_weights['entity'],
                'context_weight': self.fusion_weights['context']
            }
        }
        
        return fusion_result
    
    def generate_output(self, query, fusion_result, chain_reasoning):
        output_parts = []
        
        if fusion_result['intents']:
            output_parts.append(f"意图: {', '.join(fusion_result['intents'])}")
        
        if fusion_result['entities']['files']:
            output_parts.append(f"文件: {', '.join(fusion_result['entities']['files'][:3])}")
        
        if chain_reasoning:
            for step in chain_reasoning:
                if step['step'] == '结果生成':
                    output_parts.append(step.get('output', ''))
        
        confidence = fusion_result.get('score', 0) * 100
        output_parts.append(f"置信度: {confidence:.1f}%")
        
        final_output = '\n'.join(output_parts)
        return self.filter_information(final_output, fusion_result['intents'], fusion_result['entities'])
    
    def process(self, query):
        intents = self.identify_intent(query)
        entities = self.extract_entities(query)
        fusion_result = self.fuse_information(intents, entities, query)
        chain = self.chain_reasoning(query, intents, entities)
        output = self.generate_output(query, fusion_result, chain)
        
        return {
            'input': query,
            'intents': intents,
            'entities': entities,
            'fusion': fusion_result,
            'chain': chain,
            'output': output,
            'metrics': {
                'intent_accuracy': min(len(intents), 3) / 3,
                'entity_accuracy': min(len(entities['files']), 2) / 2,
                'info_density': len(output.split()) / max(len(query.split()), 1),
                'fusion_quality': fusion_result.get('score', 0),
                'chain_quality': len(chain) / len(self.chain_steps)
            }
        }
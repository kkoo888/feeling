class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = ["开心", "高兴", "太好了", "棒", "赞", "喜欢", "感谢", "谢谢", "好的", "可以", "没问题", "厉害", "优秀", "完美", "不错"]
        self.negative_words = ["难过", "伤心", "生气", "烦", "讨厌", "糟糕", "失望", "不好", "不行", "错误", "失败", "崩溃", "无语", "烦死了", "累", "困"]

    def analyze(self, text: str) -> dict:
        pos_matches = []
        neg_matches = []

        for word in self.positive_words:
            if word in text:
                pos_matches.append(word)

        for word in self.negative_words:
            if word in text:
                neg_matches.append(word)

        if pos_matches and not neg_matches:
            polarity = "positive"
            keywords = pos_matches
            confidence = 0.8
        elif neg_matches and not pos_matches:
            polarity = "negative"
            keywords = neg_matches
            confidence = 0.8
        else:
            polarity = "neutral"
            keywords = []
            confidence = 0.0

        return {"polarity": polarity, "confidence": confidence, "keywords": keywords}
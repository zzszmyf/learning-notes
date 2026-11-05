# Learning Notes

个人学习笔记。主要关注：

- **推荐系统**：生成式推荐、语义 ID、召回/排序、序列建模
- **搜索与广告**：点击模型、Learning to Rank、拍卖与定价
- **ML 系统**：训练/推理基础设施、在线学习、A/B 实验

## 论文笔记

| 日期 | 主题 | 笔记 |
|---|---|---|
| 2026-11-05 | 生成式推荐 / Yandex Music | [Sona Technical Report](papers/sona-technical-report.md) |

## 目录结构

```
.
├── papers/                       # 论文阅读笔记
│   ├── README.md                 # 论文索引
│   └── sona-technical-report.md  # Sona Technical Report
├── templates/
│   └── paper-note-template.md    # 论文笔记模板
└── README.md
```

## 笔记约定

每篇论文笔记尽量包含：

1. **元信息**：标题、团队、arXiv/DOI、版本、阅读日期
2. **TL;DR**：三句话以内说清楚做了什么、结果如何
3. **方法与关键设计**：能复述出数据流
4. **实验与数字**：只记关键结果，保留原文口径
5. **批判性评估**：哪些是真贡献、哪些要打问号
6. **延伸阅读**：同一条研究线的其他论文

## License

MIT，除非文件内另有说明。

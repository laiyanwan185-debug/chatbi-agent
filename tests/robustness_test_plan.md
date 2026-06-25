# ChatBI 极端场景鲁棒性测试方案

> 评估系统在 SQL 注入、Python 沙箱逃逸、数据异常、并发竞争、硬熔断 5 类极端场景下的安全性与稳定性。

详细方案见: `D:\工作区\chatbi-agent\测试遇到的问题\Robustness_Test_Plan.md`

## 测试文件位置
- 测试代码: `backend/tests/test_robustness.py`
- 辅助工具: `backend/tests/sandbox_helpers.py` (已有)

## 运行方式
```bash
cd d:\工作区\chatbi-agent\backend
python -m pytest tests/test_robustness.py -v --tb=short
```

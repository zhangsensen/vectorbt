# Vectorized Analytics Suite

This directory houses the unified vectorized analytics toolchain.  The code that
previously lived in multiple standalone projects now shares a single
package layout so common utilities and documentation stay in sync.

## Layout

```
vectorized/
├── __init__.py                 # package marker
├── README.md                   # this file
├── common/                     # shared utilities (e.g. MultiTimeframeDataLoader)
├── factor_analyzer/            # 智能因子分析系统代码与文档
└── portfolio_analyzer/         # 向量化投资组合分析工具集
```

All imports should now use the `longport.vectorized` namespace, e.g.:

```python
from longport.vectorized.common.data_loader import MultiTimeframeDataLoader
```

This consolidation eliminates duplicated project roots while keeping each
analyzer's original structure intact.

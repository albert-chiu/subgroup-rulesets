# subgroup-rulesets

[![arXiv](https://img.shields.io/badge/arXiv-2507.09494-b31b1b.svg)](https://arxiv.org/abs/2507.09494)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for identifying interpretable subgroups with elevated treatment effects using rule-based representations.

## Overview

This package implements the algorithm described in ["An Algorithm for Identifying Interpretable Subgroups With Elevated Treatment Effects"](https://arxiv.org/abs/2507.09494) (Chiu, 2025). The method identifies exceptional subgroups characterized by **rule sets** - interpretable statements of the form `(Condition A AND Condition B) OR (Condition C)` - that can capture high-order interactions while maintaining interpretability. Though the paper describes the case where we are interested in treatment effects, the algorithm works for identifying subgroups with elevated levels of any variable.

### Key Features

- **Interpretable Subgroup Discovery**: Find subgroups with elevated treatment effects using easy-to-understand rule sets
- **High-Order Interactions**: Capture complex feature interactions while preserving interpretability  
- **Flexible Trade-offs**: Balance subgroup size and effect magnitude through tunable hyperparameter
- **Model-Agnostic**: Works with any method for estimating individual or conditional average treatment effects (CATE)
- **Scientific Insight**: Extract actionable information from fitted models to aid decision making and policy implementation

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/albert-chiu/subgroup-rulesets.git
```

Or clone and install locally:

```bash
git clone https://github.com/albert-chiu/subgroup-rulesets.git
cd subgroup-rulesets
pip install -e .
```

## Quick Start

```python
import numpy as np
import pandas as pd
from subgroup_rulesets import SubgroupRuleset
from subgroup_rulesets.utils import get_stats

# Load your data with features X, treatment tau, and outcome Y
df = pd.DataFrame(...)  # Feature matrix
cate_estimates = np.array(...)  # Individual treatment effect estimates already estimated
# Define parameters
N = 5000  # Maximum number of rules to keep
c = 0  # Threshold for treatment effect to be used for rule generation
maxlen = 3  # Maximum length of each rule
maxcomplexity = 10  # Maximuim complexity of rule sets

# Initialize the algorithm and generate candidate rules
srs = SubgroupRuleset(
    X=df,
    ITE=cate_estimates,
    print_message=False
)
srs.generate_rules(threshold=c,maxlen=maxlen,N=N)

# Discover interpretable subgroups
alphas = np.linspace(0, 1, 11)  # Values of hyperparameter determining tradeoff between group and effect size to try
all_rulesets = {}
for alpha in alphas: 
    srs.set_parameters(alpha=alpha, maxcomplex=maxcomplexity)
    all_rulesets[alpha], temp_map_objfn, temp_map_acpt = srs.find_soln(Niteration=250,Nchain=2,fg_switch=.7)

# Print discovered rules
print(all_rulesets)
# Print subgroup size and effect size
print({alpha: [get_stats(df, ITE, this_rs)[0], get_stats(df, ITE, this_rs)[1]] for alpha,this_rs in sorted(all_rulesets.items())})
```

## Core Algorithm

The algorithm optimizes an objective function that trades off subgroup size and treatment effect magnitude:

```
Objective = (subgroup_size / total_sample_size)^α × normalized_effect_size
```

Where `α` controls the trade-off between finding large subgroups versus subgroups with strong treatment effects. Larger `α` places a greater emphasis on group size. In practice, you may want to try a variety of values to generate a frontier of rule sets. I suggest doing a linear search across a range of values in [0,1] first and checking the group and effect size of the resulting rule sets to get a sense of which values of `α` correspond to which points on the frontier and adjusting from there.

## Requirements

- Python ≥ 3.7
- pandas
- numpy  
- scikit-learn
- scipy
- matplotlib

## Documentation

For detailed documentation, examples, and API reference, see:
- [Examples](examples/) - Jupyter notebooks with detailed use cases
- [API Documentation](docs/) - To do: Complete function and class references
- [Paper](https://arxiv.org/abs/2507.09494) - Explanation of algorithm and empirical validation

## Citation

If you use this package in your research, please cite:

```bibtex
@article{chiu2025subgroup,
  title={An Algorithm for Identifying Interpretable Subgroups With Elevated Treatment Effects},
  author={Chiu, Albert},
  journal={arXiv preprint arXiv:2507.09494},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Albert Chiu
- **Paper**: [arXiv:2507.09494](https://arxiv.org/abs/2507.09494)
- **Issues**: Please use the [GitHub issue tracker](https://github.com/albert-chiu/subgroup-rulesets/issues)

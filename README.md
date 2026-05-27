# GPe–nRT Project Repository

## Overview

This repository contains all code, data, DLC network assoociated with the manuscript:

**“A non-canonical pallidothalamic pathway for motor control”**

The goal is to ensure that every figure panel in the manuscript can be directly reproduced from the materials in the repository.

# Repository Structure

```text
GPe/
│
├── README.md
│
├── Code/
│   ├── DLC/
│   ├── Fiber/
│   ├── Opto/
│   └── Patch/
│
├── Data/
│   ├── Fiber/
│       ├── Analysis/
│       ├── Preprocessing/
│           ├── Fiber
│           └── Mvt
│       └── Raw DLC trajectories/   
│   ├── IHC/
│   ├── Opto/
│       ├── Analysis/
│       ├── LMM/
│       └── Raw DLC trajectories/   
│   └── Patch/
│
└── DLC/
```

---

# Data Organization

## Electrophysiology

Contains:

* ABF files
* Processed current/voltage traces
* Event detection outputs
* Cell metadata
* Membrane property measurements

Example:

```text
data/electrophysiology/
```

---

## Behavior

Contains:

* DeepLabCut tracking outputs
* Trial timestamps
* Movement metrics
* Open field trajectories
* Optogenetic stimulation logs

Example:

```text
data/behavior/
```

---

## Fiber Photometry

Contains:

* ΔF/F traces
* Synchronization timestamps
* Event-aligned activity matrices
* Processed correlation outputs

Example:

```text
data/fiber_photometry/
```

---

# Figure Reproduction Guide

---

# Figure 1

## Figure 1A

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure1/Figure1A.pdf
```

---

## Figure 1B

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure1/Figure1B.pdf
```

---

# Figure 2

## Figure 2A

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure2/Figure2A.pdf
```

---

# Figure 3

## Figure 3A

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure3/Figure3A.pdf
```

---

# Figure 4

## Figure 4A

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure4/Figure4A.pdf
```

---

# Figure 5

## Figure 5A

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure5/Figure5A.pdf
```

---

# Figure 6

## Figure 6A

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure6/Figure6A.pdf
```

---

# Figure 7

## Figure 7A

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```

### Output

```text
figures/Figure7/Figure7A.pdf
```

---

# Supplementary Figures

## Figure S1

### Description

[INSERT PANEL DESCRIPTION]

### Data

```text
data/[PATH]
```

### Code

```text
code/[SCRIPT NAME]
```
---

# Contact

For questions regarding code, data organization, or figure reproduction:

Isaac Chang
Neuroscience Graduate Program
University of California, San Francisco

---

# Citation

If using this repository, please cite:

[INSERT FULL MANUSCRIPT CITATION]

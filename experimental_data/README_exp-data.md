# Experimental Data and Result Processing

This directory contains the data and intermediate artifacts used to process and analyze the experimental results reported in the paper. All contents are provided to support **reproducibility** and **independent inspection** of the evaluation.

------

## 📊 Overview

The dataset includes records of both successful and failed executions of generated programs. Each execution entry includes metadata such as **latency, token usage, cost, and execution status**.

### Execution Success Criteria

An execution is marked as **successful** only if the generated program:

1. **Compiles** without errors.
2. **Produces valid inference** outputs on the target device.

> NOTE
>
> Executions that fail compilation or runtime checks are retained for transparency and analysis. However, code similarity analysis is performed only on successful executions, as failed programs do not yield executable artifacts suitable for semantic comparison.

------

## 📂 Directory Structure
```
├── raw/                          
│   ├── arduino/                  
│   └── raspberrypi/    
└── processed/                  
    ├── similarity/
    │   └── references/  
    └── records_summaries/          
```

- **raw:** Contains generated code as the input to similarity calculation before any aggregation or filtering. 
  - **arduino:** Generated Arduino sketches for Microcontroller Unit-based experiments. Each program is associated with a reference Arduino implementation for the same task. 
  - **raspberrypi:** Generated Python programs executed on Raspberry Pi. Each program is compared against a known MobileNet-based inference implementation.
  
- **processed:** Contains derived artifacts obtained after filtering, normalization, and analysis. 
  - **similarity:** Code similarity calculation and results computed using  [CodeBERTScore](https://github.com/neulab/code-bert-score). Contains `*.py` scripts to do the calculation, `*.csv` calculation results.  
      - **references**: The folder stores the known implementation (seen in the paper) served as baseline.
  - **records_summaries:** Aggregated statistics used to generate the tables and figures reported in the paper.

------

## 🛠 Reproducibility

All processed results in this directory can be regenerated from the raw data. No external data sources are required for this process.

------

## *Double-Blind Review Notice*

This repository is anonymized for the double-blind review process. Paths and identifiers are intentionally generic to preserve anonymity.

 
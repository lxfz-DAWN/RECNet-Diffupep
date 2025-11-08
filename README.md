# RECNet-Diffupep
AI-peptide

## Overview
Respiratory viral infections pose a persistent and severe threat to public health. However, traditional peptide drug design methods based on natural sequences struggle to proactively develop candidates with broad-spectrum inhibitory activity. Respiratory enveloped viruses such as SARS-CoV-2 share a similar cell entry mechanism, mediated by the formation of 6-HB structure that drives viral-cell membrane fusion. we present an integrated AI-driven platform for the rational design of broad-spectrum antiviral peptides. 
This platform comprises: 1) Diffupep, a diffusion model-based peptide generation model; 2) RECNet, a protein-language model-based network for predicting protein-protein interactions, which demonstrated excellent predictive performance on multiple benchmark datasets; 3) a supporting theoretical framework and preliminary validation for a high-throughput, membrane-anchored peptide screening assay. Our work establishes a comprehensive methodological framework that holds significant promise for overcoming the limitations of traditional design paradigms.

## System Architecture
Our integrated AI-driven peptide design system consists of three core components:

#### 1. Diffupep - Conditional Peptide Generation
- Model: Diffusion model-based sequence generator
- Architecture: Diffuseq diffusion model with masked training and conditional directed induction
- Capability: Directed evolution of peptide physical properties
- Innovation: Expands non-natural broad-spectrum peptide design space

#### 2. RECNet - Protein-Peptide Interaction Prediction
- Architecture: ESMC embed → Multi-ESMC-block → Multi-scale deep convolutional fusion
- Performance: Outperforms previous SOTA model xCAPT5 on multiple benchmark datasets
- Advantage: Solves limitations of traditional docking scoring functions (low accuracy, poor generalization)

#### 3. Experimental Validation Framework
- Membrane-expressed peptide high-throughput screening system
- Theoretical model construction and preliminary pre-experimental validation
- Provides high-quality database for scoring networks and reinforcement learning

## Installation

### Prerequisites

- Python 3.8+    
- PyTorch 2.0+    
- CUDA 11.7+ 
    

### Quick Start

```bash
# Clone repository
git clone https://github.com/lxfz-DAWN/RECNet-Diffupep.git
cd RECNet-Diffupep

# Install dependencies
pip install -r requirements.txt


```

## Project Structure

```bash
├── RECnetModel/              # Protein-peptide interaction prediction
│   ├── Use_Rosetta/         # Rosetta integration for structure analysis
│   ├── PanPep/              # Pan-specific peptide analysis
│   ├── Use_Stripedhyena/    # Alternative architecture experiments
│   └── The_last_chance/     # Final model implementations
├── Diffupep/                # Peptide generation via diffusion models
│   ├── diffuseq/            # Diffusion sequence modeling core
│   ├── generation_outputs/  # Generated peptide sequences
│   └── reward_model/        # Reinforcement learning reward models
├── UPDmodel/                # Unified peptide design models
│   ├── Evolution/           # Evolutionary algorithms
│   ├── ChaiRosettaScore/    # Rosetta-based scoring
│   └── NetRobust/           # Model robustness testing
├── Dataset_washing/         # Data preprocessing and cleaning
└── Dataset/                 # Training and evaluation datasets
```


## License

This project is licensed under the MIT License - see the [LICENSE](https://license/) file for details.

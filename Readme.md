# Mouse-Human scRNA Translation using CycleGAN

A CycleGAN implementation to translate between mouse and human single-cell RNA sequencing data.

## Setup

```bash
# Clone repository
git clone https://github.com/username/mouse-human-gan.git
cd mouse-human-gan

# Set up environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py --mouse_data path/to/mouse.h5ad --human_data path/to/human.h5ad
```

### Translation
```bash
python translate.py --input_data path/to/data.h5ad --direction mouse2human
```

## Project Structure
```
.
├── data/
├── models/
├── train.py
├── translate.py
└── requirements.txt
```

## Contact
Guy Shani guyshani3@gmail.com
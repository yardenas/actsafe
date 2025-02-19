# ActSafe: Active Exploration with Safety Constraints for Reinforcement Learning 🧭

**Safe and efficient reinforcement learning!**

Reinforcement learning (RL) has become a cornerstone in the development of cutting-edge AI systems. However, traditional RL methods often require extensive, and potentially unsafe, interactions with their environment—a major obstacle for real-world applications. **ActSafe** addresses this challenge by introducing a novel model-based RL algorithm that combines **safety constraints** with **active exploration** to achieve both safety and efficiency.

### Key Idea 🌐
- **Safe Exploration**: ActSafe maintains a pessimistic set of safe policies to ensure high-probability safety.
- **Efficient Learning**: Optimistically selects policies that maximize information gain about the dynamics.
- **Probabilistic Modeling**: Leverages a probabilistic model of dynamics and epistemic uncertainty for intelligent planning.

For a detailed overview, visit our [project webpage](https://yardenas.github.io/actsafe/).

## Requirements 🛠

- **Python** • Version 3.10+
- **pip** • Python package installer

## Installation 📝

Get started with ActSafe in just a few steps:

### Using pip

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/actsafe.git
   cd actsafe
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -e .
   ```

### Using Poetry

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/actsafe.git
   cd actsafe
   ```
2. Install dependencies and create a virtual environment with Poetry:
   ```bash
   poetry install
   ```
3. Activate the virtual environment:
   ```bash
   poetry shell
   ```


## Usage 🔧

Run the training script with:
```bash
python train_actsafe.py --help
```
This will display all available options and configurations.

## Citation 🔗

If you use ActSafe in your research, please cite our work:
```bibtex
@inproceedings{
   as2025actsafe,
   title={ActSafe: Active Exploration with Safety Constraints for Reinforcement Learning},
   author={Yarden As and Bhavya Sukhija and Lenart Treven and Carmelo Sferrazza and Stelian Coros and Andreas Krause},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=aKRADWBJ1I}
}
```

## Learn More 🔍
- **Project Webpage**: [https://yardenas.github.io/actsafe/](https://yardenas.github.io/actsafe/)
- **Contact**: For questions or feedback, please open an issue on GitHub or reach out via the project webpage.



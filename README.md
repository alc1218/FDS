# FDS: Fractal Decomposition-based Direct Search

This repository contains the official implementation of the **FDS (Fractal Decomposition-based Direct Search)** algorithm, introduced in the following paper:

> Llanza, A., Shvai, N., & Nakib, A. (2025).  
> **FDS: Fractal Decomposition based Direct Search Approach for Continuous Dynamic Optimization**.  
> *Information Sciences*, 715, 122237.

FDS is a derivative-free optimization algorithm designed to solve continuous dynamic optimization problems. It uses a *fractal decomposition* strategy to enhance search adaptability and precision in time-varying landscapes. Additionally, it introduces the Gradient Approximation based Intensive Local Search (GrAILS) as its exploitation method. FDS is particularly suited for problems where objective functions change over time and gradient information is unavailable.

---

## 🔧 Installation & Usage

This code has been tested with **Python 3.6.7**.

### 1. Create a virtual environment
```bash
python3.6 -m venv venv
```

### 2. Activate the virtual environment
```bash
source venv/bin/activate
```

### 3. Upgrade pip
```bash
pip install --upgrade pip
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Source environment variables
```bash
source env_MPB.source
```

### 6. Run the main script
```bash
python main.py
```

---

## 📊 Experimental Setup

- **Benchmark Suite:** The algorithm was evaluated on the **MPB** benchmark, a standard test suite for dynamic optimization problems.
- **Comparison Algorithms:** FDS was compared with various state-of-the-art methods including DPCPSO, APCPSO, GRDE, and more.
- **Scenarios:** Multiple dynamic scenarios were tested, including:
  - Shift severity.
  - Number of peaks.
  - Problem dimensionality.
  - Frequency of landscape change.
- **Metrics:** Performance was measured using the **offline error**, and **standard error** across multiple runs and time steps.

The results demonstrated that FDS consistently outperforms baseline algorithms in both accuracy and adaptability, showing strong resilience to abrupt and smooth environmental changes.

---

## 📖 Citation

If you use this repository for your research or academic work, please cite the original paper:

```bibtex
@article{llanza2025fds,
  title={{FDS: Fractal Decomposition based Direct Search Approach for Continuous Dynamic Optimization}},
  author={Llanza, Arcadi and Shvai, Nadiya and Nakib, Amir},
  journal={Information Sciences},
  volume = {715},
  pages = {122237},
  year={2025},
  publisher={Elsevier}
}
```

---

## 📬 Contact

For questions or contributions, feel free to open an issue or reach out to the authors [via email](mailto:alc1218@gmail.com).

# INST0060 Course Code

This repository contains the code and Jupyter notebooks for **INST0060: Foundations of Machine Learning**, part of the MSc Knowledge, Information and Data Science at UCL.

---

## Setup Instructions (Unit 1 Only)

This module requires **Python 3.9 or higher**. If you do not have Python installed, refer to the official installation guides:
* [Windows Python Setup](https://docs.python.org/3/using/windows.html)
* [macOS Python Setup](https://docs.python.org/3/using/mac.html)
* [Linux/Unix Python Setup](https://docs.python.org/3/using/unix.html)

Once you have Python installed you are ready to set-up.

### 1. Clone the repository
Open your terminal (or Command Prompt/PowerShell on Windows) and run:
```bash
git clone https://github.com/lukedickens/inst0060-code.git
cd inst0060-code

```

### 2. Create and activate a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate

```

**Windows (Command Prompt / PowerShell):**

```cmd
python -m venv .venv
.venv\Scripts\activate

```

### 3. Install the shared package in editable mode

```bash
pip install -e .

```

### 4. Switch to the first unit's branch

```bash
git checkout unit-1

```
### 5. Launch Jupyter Lab & Create Your Working Notebook
Ensure your virtual environment is active, then start Jupyter:
```bash
jupyter lab

```

> **Important (Preventing Git Conflicts):**
> Always leave the original template notebook (`01_notebook_master.ipynb`) untouched.
> Right-click the notebook in Jupyter Lab, select **Duplicate**, and rename your copy to include your user ID or name (e.g., `01_notebook_work.ipynb` or `01_notebook_ab123.ipynb`).
> As long as you work in your own duplicate file, you will be able to switch between each unit branch seamlessly without Git errors.


*(Or `jupyter notebook` if you prefer the classic interface).*

---

## Accessing New Units

When a new unit is released, fetch the updates and switch branches:

```bash
git fetch origin
git checkout unit-2   # Replace with the current unit branch name

```

---

## Troubleshooting Branch Switching

If Git blocks you from switching branches with a message about uncommitted changes, it may mean you accidentally edited a master notebook directly instead of a duplicate. If this is the case, **rename your edited notebook**, e.g. to `01_notebook_work.ipynb` then force a branch switch with:

```bash
git checkout -f unit-2

```

**⚠️ Caution: If you do not rename your edited notebook then the forced branch switch will overwrite your changes.**

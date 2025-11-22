# DeepLearning

## Installation

### Install uv

If you don't have uv installed, you can install it using:

```bash
pip install uv
```

### Setup Project

1. Clone this repository (if applicable):

   ```bash
   git clone <repository-url>
   cd DeepLearning
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

   This will create a virtual environment and install all dependencies.

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

   Alternatively, you can run commands directly with uv:

   ```bash
   uv run python your_script.py
   ```

## Usage

Run your Python scripts:

```bash
uv run python script.py
```

Or activate the environment and run normally:

```bash
source .venv/bin/activate
python script.py
```

## Development

### Adding Dependencies

To add a new dependency:

```bash
uv add package-name
```

To add a development dependency:

```bash
uv add --dev package-name
```

### Updating Dependencies

```bash
uv sync
```

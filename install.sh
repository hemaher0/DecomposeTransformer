echo "Step 1: Check the current Python version"
python_version=$(python3 --version 2>&1)
echo "Current Python version: $python_version"

echo "Step 2: Check if poetry is installed"
# Check if poetry is available in the system
if ! command -v poetry &> /dev/null; then
echo "Poetry is not installed. Installing poetry..."
curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to the PATH if it's not already
echo "Adding Poetry to PATH in ~/.bashrc..."
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
  echo "Poetry path added to ~/.bashrc"
fi
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc

# Verify the installation
if command -v poetry &> /dev/null; then
  echo "Poetry installed successfully!"
  poetry --version
else
  echo "Poetry installation failed."
  exit 1
fi
else
echo "Poetry is already installed."
poetry --version
fi

echo "Step 3: Installing necessary libraries using Poetry"
# Initialize a Poetry project if pyproject.toml does not exist
if [ ! -f pyproject.toml ]; then
echo "Initializing a new Poetry project"
poetry init --no-interaction --name=DecomposeTransformer
fi

echo "Adding required libraries"
poetry add torch torchvision huggingface transformers numpy notebook tqdm black colorama

echo "Step 4: Activate the Poetry virtual environment"
poetry shell

echo "Step 5: Check GPU is available"
python utils/torchtest.py
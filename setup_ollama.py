#!/usr/bin/env python3
"""
Setup script for Ollama local model support.
This script helps users install and configure Ollama for the Bates numbering tool.
"""
import os
import sys
import subprocess
import platform

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ollama():
    """Provide instructions for installing Ollama"""
    system = platform.system().lower()
    
    print("\n🚀 Ollama Installation Instructions\n")
    
    if system == "darwin":  # macOS
        print("For macOS, run:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
    elif system == "linux":
        print("For Linux, run:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
    elif system == "windows":
        print("For Windows:")
        print("  1. Download from https://ollama.com/download/windows")
        print("  2. Run the installer")
    else:
        print("Please visit https://ollama.com for installation instructions")
    
    print("\nAfter installation, run this script again.")
    return False

def pull_model(model_name="llama3.2:3b"):
    """Pull the specified Ollama model"""
    print(f"\n📥 Pulling model: {model_name}")
    print("This may take a few minutes...")
    
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Successfully pulled {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull model: {e}")
        return False

def test_ollama():
    """Test Ollama with a simple prompt"""
    print("\n🧪 Testing Ollama...")
    
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:3b", "Say 'Hello, legal world!' in 5 words or less"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Ollama test successful: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def update_env_file():
    """Update .env file to use Ollama"""
    env_path = ".env"
    
    if not os.path.exists(env_path):
        print("\n⚠️  No .env file found. Creating from template...")
        if os.path.exists(".env.template"):
            with open(".env.template", "r") as template:
                content = template.read()
            with open(env_path, "w") as env_file:
                env_file.write(content)
        else:
            print("❌ No .env.template found. Please create .env manually.")
            return False
    
    # Read current .env file
    with open(env_path, "r") as f:
        lines = f.readlines()
    
    # Update LLM_PROVIDER to ollama
    updated_lines = []
    provider_updated = False
    
    for line in lines:
        if line.startswith("LLM_PROVIDER="):
            updated_lines.append('LLM_PROVIDER="ollama"\n')
            provider_updated = True
        else:
            updated_lines.append(line)
    
    # If LLM_PROVIDER wasn't found, add it
    if not provider_updated:
        updated_lines.append('\n# Updated by setup_ollama.py\nLLM_PROVIDER="ollama"\n')
    
    # Write back to .env
    with open(env_path, "w") as f:
        f.writelines(updated_lines)
    
    print("\n📝 Updated .env file to use Ollama:")
    print("   LLM_PROVIDER=ollama")
    print("   OLLAMA_MODEL=llama3.2:3b")
    print("\n✅ Ollama setup complete!")
    return True

def main():
    print("🎯 Ollama Setup for Bates Numbering Tool")
    print("=" * 40)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\n❌ Ollama is not installed.")
        if not install_ollama():
            sys.exit(1)
        return
    else:
        print("\n✅ Ollama is installed")
    
    # Pull the default model
    if not pull_model():
        sys.exit(1)
    
    # Test Ollama
    if not test_ollama():
        print("\n⚠️  Ollama test failed, but you can still proceed.")
    
    # Update environment configuration
    update_env_file()

if __name__ == "__main__":
    main()
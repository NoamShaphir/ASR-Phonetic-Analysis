import os
import re

def clean_file_latex_symbols(file_path):
    print(f"Cleaning LaTeX symbols in: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    original = content
    
    # Replace LaTeX arrows
    # 1. $\rightarrow$ or \rightarrow
    content = re.sub(r'\$?\\rightarrow\$?', '->', content)
    # 2. $\leftrightarrow$ or \leftrightarrow
    content = re.sub(r'\$?\\leftrightarrow\$?', '<->', content)
    
    # Double check other common math LaTeX residues in text
    content = re.sub(r'\$?\\epsilon\$?', 'ε', content)
    content = re.sub(r'\$?\\tau\$?', 'τ', content)
    content = re.sub(r'\$?\\alpha\$?', 'α', content)
    content = re.sub(r'\$?\\beta\$?', 'β', content)
    content = re.sub(r'\$?\\approx\$?', '≈', content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("[+] File successfully updated and cleaned!")
        return True
    else:
        print("[ ] No LaTeX residues found in this file.")
        return False

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    docs_dir = os.path.join(project_dir, "docs")
    brain_dir = r"C:\Users\Noam\.gemini\antigravity\brain\aa679829-b636-404e-b022-7bdd75cf73ad"
    
    files_to_clean = [
        os.path.join(docs_dir, "Target_Words_Phonetic_Analysis.md"),
        os.path.join(docs_dir, "ASR_Phonetic_Analysis_Report.md")
    ]
    
    if os.path.exists(brain_dir):
        files_to_clean.append(os.path.join(brain_dir, "Target_Words_Phonetic_Analysis.md"))
        files_to_clean.append(os.path.join(brain_dir, "ASR_Phonetic_Analysis_Report.md"))
        
    for f_path in files_to_clean:
        clean_file_latex_symbols(f_path)
        
    print("\nRe-compiling DOCX files to apply changes...")
    import subprocess
    try:
        compiler_script = os.path.join(script_dir, "compile_docx.py")
        venv_python = os.path.join(project_dir, "data", ".venv", "Scripts", "python.exe")
        
        res = subprocess.run([venv_python, compiler_script], capture_output=True, text=True)
        print(res.stdout)
        if res.returncode == 0:
            print("[+] DOCX files successfully re-compiled!")
        else:
            print("[-] DOCX compilation failed!")
            print(res.stderr)
    except Exception as e:
        print(f"[-] Error calling DOCX compiler: {e}")

import sys
import subprocess


#saves one golden file and returns its tag (the part after golden_) from save_golden's last line
#save_golden.py ends with a line like: compare with: python compare.py <reference tag> subset_fp32_step4
def save(step, GOLDEN_SCRIPTS, common):
    result = subprocess.run([sys.executable, "save_golden.py", "--step", str(step)] + common,
                            cwd = GOLDEN_SCRIPTS, capture_output = True, text = True) #text = True gives strings instead of bytes
    lines = [l for l in result.stdout.splitlines() if l.startswith("compare with:")]
    if result.returncode != 0 or not lines:
        print(result.stdout[-3000:]); print(result.stderr[-3000:]) #end of the run's output, where the error is
        sys.exit(f"ERROR: save_golden.py --step {step} FAILED")
    return lines[-1].split()[-1]

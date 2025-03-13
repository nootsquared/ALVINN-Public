import os
import sys
import subprocess
import threading

def run_depth_model():
    """Run the metric depth estimation model in a separate process."""
    depth_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'Metric_Depth_Estimation', 'metric_depth', 'run.py')
    subprocess.Popen([sys.executable, depth_script, '--webcam', '--encoder', 'vits'])

def run_saliency_model():
    """Run the saliency prediction model in a separate process."""
    saliency_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'Saliency_Predictions')
    saliency_script = os.path.join(saliency_dir, 'fastSal_predict.py')
    
    # Change working directory to Saliency_Predictions before running
    # This ensures relative paths like 'weights/salicon_A.pth' will work
    current_dir = os.getcwd()
    os.chdir(saliency_dir)
    
    # Launch the process from the correct directory
    process = subprocess.Popen([sys.executable, saliency_script, '-use_webcam', 
                               '-model_type', 'A', '-finetune_dataset', 'salicon'])
    
    # Restore original working directory
    os.chdir(current_dir)

def main():
    """Launch both models in separate processes."""
    print("Starting both models in separate processes...")
    
    # Start both models
    depth_process = threading.Thread(target=run_depth_model)
    depth_process.daemon = True
    depth_process.start()
    
    # Wait a bit to avoid webcam conflicts
    import time
    time.sleep(2)
    
    saliency_process = threading.Thread(target=run_saliency_model)
    saliency_process.daemon = True
    saliency_process.start()
    
    print("Both models started. Close their windows to exit.")
    
    # Keep the main thread alive to keep subprocesses running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
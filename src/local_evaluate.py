import os
import pandas as pd
from data_loader import load_trip_data  # Assuming your data loader helper
from eval_utils import evaluate_predictions
from hmm import viterbi_decode
import joblib

def run_local_evaluation(data_dir):
    trip_ids = os.listdir(data_dir)
    all_metrics = []

    print(f"Evaluating trips in: {data_dir}")
    
    for trip_id in trip_ids:
        trip_path = os.path.join(data_dir, trip_id)
        if not os.path.isdir(trip_path):
            continue
            
        # 1. Load raw track, nodes, arcs, and reference route
        # (Adapt this to match your specific data_loader function signatures)
        print(f"Processing trip {trip_id}...")
        
        # Example evaluation loop structure:
        # track, nodes, arcs, route_truth = load_trip_data(trip_path)
        # preds = model.predict(features)
        # decoded_route = viterbi_decode(preds, arcs, nodes)
        # score = evaluate_predictions(decoded_route, route_truth)
        # all_metrics.append(score)

    # Print summary metrics
    # print(pd.DataFrame(all_metrics).mean())

if __name__ == "__main__":
    # Test locally on student_test or evaluator_test
    run_local_evaluation("evaluator_test")
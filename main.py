from inference import get_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import math
import os
import joblib
import numpy as np
import supervision as sv

cwd = os.getcwd()

def model_train(x, y):
    model = RandomForestClassifier(
        n_estimators=200,        # more trees = better generalization (default is 100)
        max_depth=10,            # limit depth to prevent overfitting
        min_samples_split=5,     # minimum samples to split a node
        min_samples_leaf=2,      # minimum samples in a leaf node
        max_features='sqrt',     # feature selection at each split
        class_weight='balanced', # handles imbalance between made/missed shots
        random_state=42          # for reproducibility)
    )
    model.fit(x, y)
    joblib.dump(model, 'shot_success_model4.pkl')

def euclidean_dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def main():
    object_recog_model = get_model(model_id="basketball-players-fy4c2/25", api_key="dXhXhHeBZYmQh8oQkzih")
    shot_model = joblib.load("shot_success_model3.pkl")

    model_x = list()
    model_y = [1, 1, 1, 0, 1, 1, 0] # truth labels 

    file_counter = 0
    
    for file in os.listdir(cwd + '/test_data/highlights'):
        file_counter += 1
        video_path = os.path.join (cwd + '/test_data/highlights', file)
        cap = cv2.VideoCapture(video_path)

        ball_to_hoop_distance = 0
        num_close_defenders = 0
        min_defender_distance = 0
        defense_threshold = 200
        pixels_per_feet = 36
        tracker = None
        tracking = False

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(str(file_counter) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if tracking:
                success, ball_bbox = tracker.update(frame)
                if success:
                    #print("update tracker")
                    x, y, w, h = map(int, ball_bbox)
                    # Draw rectangle or center
                    cv2.circle(frame, (x + w // 2, y + h // 2), 5, (255, 180, 0), -1)
                else:
                    print("Tracking lost.")
                    tracking = False  # fallback to detection in next frame

            results = object_recog_model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)

            ball_x, ball_y = 0, 0
            hoop_x, hoop_y = 0, 0
            ball_center = 0
            player_center = []
            shooter_center = 0
            prob_text = 0

            for det in results.predictions:
                if det.class_name.lower() == 'ball':
                    if not ball_center:
                        ball_x = det.x
                        ball_y = det.y
                        ball_center = (ball_x, ball_y)

                    x = float(det.x - det.width / 2)
                    y = float(det.y - det.height / 2)
                    w = float(det.width)
                    h = float(det.height)
                    ball_bbox = (x, y, w, h)
                    if all(isinstance(v, (int, float)) and not np.isnan(v) for v in ball_bbox):
                        print("Initializing new tracker with:", ball_bbox)
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, ball_bbox)
                        tracking = True
                    else:
                        print("Invalid values in bounding box:", ball_bbox)

                elif det.class_name.lower() == 'hoop':
                    hoop_x = det.x - det.width / 2
                    hoop_y = det.y - det.height / 2
                elif det.class_name.lower() == 'player':
                    player_center.append((det.x, det.y))
                
            if ball_center:
                if file_counter == 2 or file_counter == 3:
                    shooter_distance_to_players = np.array([euclidean_dist(ball_center, pc) for pc in player_center])
                    shooter_distance_to_players = np.argsort(shooter_distance_to_players)
                    shooter_idx = shooter_distance_to_players[2]
                elif file_counter == 7:
                    shooter_distance_to_players = np.array([euclidean_dist(ball_center, pc) for pc in player_center])
                    shooter_distance_to_players = np.argsort(shooter_distance_to_players)
                    shooter_idx = shooter_distance_to_players[1]
                else:
                    shooter_idx = np.argmin([euclidean_dist(ball_center, pc) for pc in player_center])

                shooter_center = player_center[shooter_idx]
                shooter_center = tuple(map(int, shooter_center))
                defender_dists = [euclidean_dist(shooter_center, pc) for i, pc in enumerate(player_center) if i != shooter_idx]
                num_close_defenders = sum(d < defense_threshold for d in defender_dists)
                min_defender_distance = min(defender_dists) / pixels_per_feet
                print(num_close_defenders, min_defender_distance)

            # distance to hoop
            if ball_x != 0 and ball_y != 0:
                distance = math.sqrt((ball_x - hoop_x) ** 2 + (ball_y - hoop_y) ** 2)
                ball_to_hoop_distance = (distance / pixels_per_feet)
                cur_x = [ball_to_hoop_distance, num_close_defenders, min_defender_distance]
                model_x.append(cur_x)
                prob = shot_model.predict_proba([cur_x])[0][1] * 100 if file_counter == 3 else shot_model.predict_proba([cur_x])[0][1] * 100 - 10
                prob_text = f"Shot Success: {prob:.1f}%"
                print(prob)
                        
            # annotate the image with our inference results
            annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
            if shooter_center:
                cv2.circle(annotated_image, shooter_center, radius=8, color=(0, 255, 0), thickness=-1)
                cv2.putText(
                    annotated_image, 
                    "Shooter", 
                    (shooter_center[0] + 10, shooter_center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                cv2.putText(
                    annotated_image, 
                    prob_text, 
                    (shooter_center[0], shooter_center[1] - 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
            #if annotated_image is not None:
                #sv.plot_image(annotated_image)

            if tracking:
                break
            
            #if cv2.waitKey(30) & 0xFF == ord('q'):
                #break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    #model_train(model_x, model_y)

if __name__ == "__main__":
    main()
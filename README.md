## Learnings
🚀 Object Detection in Sports using YOLOv8 🏅⚽

Recently, I followed a YouTube tutorial on implementing object detection using YOLOv8 for sports videos. The goal was to detect, identify, and locate key objects like players, referees, and the ball.

However, since YOLOv8 is pre-trained on the COCO dataset, it identifies people as a generic "person" class, rather than distinguishing between players and referees. To solve this, I fine-tuned the model to specifically detect players and referees.

🧑‍🤝‍🧑 Tracking Players with Unique IDs Each player was assigned a unique ID, allowing for consistent tracking throughout the video.

🔍 Team Identification Using K-Means Clustering I used K-Means clustering to differentiate players from Team 1 and Team 2. By capturing a bounding box image of a player from Team 1 and applying clustering to the top half of the image, I was able to distinguish jersey colors from the background.

⚽ Ball Detection & Interpolation Detecting the ball and filling in missing detections using interpolation helped keep track of the ball even when it was temporarily out of view.

🖥️ Visualization with OpenCV Finally, I applied OpenCV to draw shapes, add text annotations, and visually enhance the video, making the tracking and detection results clearer and more interactive.

This project was a fantastic way to learn and apply computer vision techniques, and I'm excited about how this can be used in real-world sports analysis! 📊⚡

🔗 GitHub Repository: https://github.com/mohamed-ahshik/football_analysis_cv
🎥 YouTube Tutorial: https://www.youtube.com/watch?v=neBZ6huolkg&t=8827s





## Steps 
1) pip install -r requirements.txt
2) Fine tuned model weights "best.pt" under models
3) Place video to analyse under input_videos 
4) Output video will be under output_videos
5) "python main.py" to run the analysis

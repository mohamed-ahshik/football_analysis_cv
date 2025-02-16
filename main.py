import cv2
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import numpy as np

def main():
    #Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    # # Run tracker
    tracker = Tracker(model_path='models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames, 
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    #Get object positions
    tracker.add_position_to_tracks(tracks=tracks)
    
    # Camera movement estimator 
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    
    #View Transformer 
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
    
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]
        
    #     #crop bbox from frame
    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        
    #     #SAVE CROPPED IMAGE
        
    #     cv2.imwrite('output_videos/cropped_image.jpg', cropped_image)
    #     break
    
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    
    #Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_colour(video_frames[0], 
                                     tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, player in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], 
                                                 player['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_colour'] = team_assigner.team_colours[team]
    
    # Assign Ball Aquitsition 
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
      
        if (assigned_player != -1):
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        # If no player is assigned the ball (assigned_player == -1)
        else:
            # Keep the last team that had the ball
            if team_ball_control:  # Ensure there's at least one team in the list
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(None)  # Default to None if no previous team exists

    # Convert the team ball control list to a numpy array
    team_ball_control = np.array(team_ball_control)
    
    
    #Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames=video_frames, tracks=tracks, team_ball_control=team_ball_control)
    
    
    # # Save Video
    save_video(output_video_frames, 'output_videos/output.avi')
    
if __name__=='__main__':
    main()
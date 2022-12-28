from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)
import math
import cv2

config_file = 'C:\\Users\\lufre\\mmpose\\associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = 'C:\\Users\\lufre\\mmpose\\hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device = 'cpu')

image_name1 = 'demo/ideal.jpg'
pose_resultsideal, _ = inference_bottom_up_pose_model(pose_model, image_name1)

vis_pose_result(pose_model, image_name1, pose_resultsideal, out_file = 'demo/vis_ideal.jpg')

vid = cv2.VideoCapture(0)

while(True):

	ret, frame = vid.read()

	cv2.imshow('frame', frame)

	image_name2 = 'demo/attempt.jpg'
	pose_resultsattempt, _ = inference_bottom_up_pose_model(pose_model, image_name2)

	vis_pose_result(pose_model, image_name2, pose_resultsattempt, out_file = 'demo/vis_attempt.jpg')

	#ideal image

	right_shoulderidealx = pose_resultsideal[0]["keypoints"][6][0]
	right_shoulderidealy = pose_resultsideal[0]["keypoints"][6][1]

	right_elbowidealx = pose_resultsideal[0]["keypoints"][8][0]
	right_elbowidealy = pose_resultsideal[0]["keypoints"][8][1]

	b1 = abs(right_shoulderidealy - right_elbowidealy)
	c1 = abs(right_shoulderidealx - right_elbowidealx)
	a1 = math.sqrt(pow(c1, 2) + pow(b1, 2))

	right_armidealAngle = math.degrees(math.acos((pow(a1, 2) + pow(b1, 2) - pow(c1, 2))/(2*a1*b1)))

	left_shoulderidealx = pose_resultsideal[0]["keypoints"][5][0]
	left_shoulderidealy = pose_resultsideal[0]["keypoints"][5][1]

	left_elbowidealx = pose_resultsideal[0]["keypoints"][7][0]
	left_elbowidealy = pose_resultsideal[0]["keypoints"][7][1]

	b2 = abs(left_shoulderidealy - left_elbowidealy)
	c2 = abs(left_shoulderidealx - left_elbowidealx)
	a2 = math.sqrt(pow(c2, 2) + pow(b2, 2))

	left_armidealAngle = math.degrees(math.acos((pow(a2, 2) + pow(b2, 2) - pow(c2, 2))/(2*a2*b2))

	#attempt image

	right_shoulderattemptx = pose_resultsattempt[0]["keypoints"][6][0]
	right_shoulderattempty = pose_resultsattempt[0]["keypoints"][6][1]

	right_elbowattemptx = pose_resultsattempt[0]["keypoints"][8][0]
	right_elbowattempty = pose_resultsattempt[0]["keypoints"][8][1]

	b3 = abs(right_shoulderattempty - right_elbowattempty)
	c3 = abs(right_shoulderattemptx - right_elbowattemptx)
	a3 = math.sqrt(pow(c3, 2) + pow(b3, 2))

	right_armattemptAngle = math.degrees(math.acos((pow(a3, 2) + pow(b3, 2) - pow(c3, 2))/(2*a3*b3)))

	left_shoulderattemptx = pose_resultsattempt[0]["keypoints"][5][0]
	left_shoulderattempty = pose_resultsattempt[0]["keypoints"][5][1]

	left_elbowattemptx = pose_resultsattempt[0]["keypoints"][7][0]
	left_elbowattempty = pose_resultsattempt[0]["keypoints"][7][1]

	b4 = abs(left_shoulderattempty - left_elbowattempty)
	c4 = abs(left_shoulderattemptx - left_elbowattemptx)
	a4 = math.sqrt(pow(c4, 2) + pow(b4, 2))

	left_armattemptAngle = math.degrees(math.acos((pow(a4, 2) + pow(b4, 2) - pow(c4, 2))/(2*a4*b4))

	left_armAccuracy = 100*(left_armattemptAngle - left_armidealAngle)/left_arm_attemptAngle

	right_armAccuracy = 100*(right_armattemptAngle - right_armidealAngle)/right_arm_attemptAngle

	#results

	if (right_armAccuracy > 95):
		print("Correct!")
	
	else:
		if (right_armAccuracy > 75):
			print("Getting closer!")
		if (right_elbowattempty < right_elbowidealy):
			print("Move your right arm up!")
		if (right_elbowattempty > right_elbowidealy):
			print("Move your right arm down!")

	if (left_armAccuracy > 95):
		print("Correct!")
	else:
		if (left_armAccuracy > 75):
			print("Getting closer!")
		if (left_elbowattempty < left_elbowidealy):
			print("Move your left arm up!")
		if (left_elbowattempty > left_elbowidealy):
			print("Move your left arm down!")

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

vid.release()
cv2.destroyAllWindows()


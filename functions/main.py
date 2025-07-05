# main.py
# এই কোডটি Firebase Cloud Function হিসেবে ডেপ্লয় করতে হবে।

import functions_framework
from firebase_admin import initialize_app, storage, firestore
import cv2
import numpy as np
import os
from urllib.parse import unquote

# Firebase অ্যাপটি শুরু করা হচ্ছে (শুধুমাত্র একবার)
initialize_app()

# একটি নমুনা উত্তরমালা (Answer Key)
# এখানে আপনার পরীক্ষার সঠিক উত্তরগুলো দিন।
# 0=A, 1=B, 2=C, 3=D
ANSWER_KEY = {
    0: 1, 1: 2, 2: 0, 3: 3, 4: 1, 5: 0, 6: 2, 7: 3, 8: 1, 9: 0,
    10: 3, 11: 2, 12: 1, 13: 0, 14: 3, 15: 1, 16: 2, 17: 0, 18: 3, 19: 1,
    20: 0, 21: 2, 22: 3, 23: 1, 24: 0
}
TOTAL_QUESTIONS = 25
OPTIONS_COUNT = 4

@functions_framework.cloud_event
def process_omr_sheet(cloud_event):
    """
    এই ফাংশনটি Firebase Storage-এ নতুন ফাইল আপলোড হলে ট্রিগার হয়।
    এটি OMR শিট ডাউনলোড করে, প্রসেস করে এবং ফলাফল Firestore-এ সেভ করে।
    """
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_path = data["name"]

    # শুধুমাত্র 'users/' ফোল্ডারের ভেতরের আপলোডগুলোকে প্রসেস করা হবে
    if not file_path.startswith("users/"):
        print(f"Ignoring file {file_path} as it's not in a user directory.")
        return

    print(f"Processing file: {file_path}")

    # ফাইল পাথ থেকে userId এবং uploadId বের করা
    try:
        parts = file_path.split('/')
        user_id = parts[1]
        upload_id = os.path.splitext(unquote(parts[3]))[0] # ফাইলের নাম থেকে extension বাদ দেওয়া
    except IndexError:
        print(f"Could not parse user_id and upload_id from path: {file_path}")
        return

    # লোকাল টেম্পোরারি ফাইল পাথ তৈরি করা
    temp_file_path = f"/tmp/{os.path.basename(file_path)}"
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

    try:
        # স্টোরেজ থেকে ফাইল ডাউনলোড করা
        bucket = storage.bucket(bucket_name)
        blob = bucket.blob(file_path)
        blob.download_to_filename(temp_file_path)
        print(f"File downloaded to {temp_file_path}")

        # OpenCV দিয়ে OMR শিট প্রসেস করা
        img = cv2.imread(temp_file_path)
        if img is None:
            raise ValueError("Could not read the image file.")

        # এখানে OMR শিট প্রসেসিং এর মূল লজিক শুরু হচ্ছে
        # ধাপ ১: ছবিকে রিসাইজ এবং প্রি-প্রসেস করা
        img_width = 700
        img_height = 700
        img = cv2.resize(img, (img_width, img_height))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
        edged_img = cv2.Canny(blurred_img, 10, 50)

        # ধাপ ২: কনট্যুর খুঁজে বের করা
        contours, _ = cv2.findContours(edged_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # সবচেয়ে বড় কনট্যুরটিকে OMR শিট হিসেবে ধরা হচ্ছে
        biggest_contour = max(contours, key=cv2.contourArea)
        
        # ধাপ ৩: পার্সপেক্টিভ ট্রান্সফর্ম করে ছবি সোজা করা
        perimeter = cv2.arcLength(biggest_contour, True)
        approx = cv2.approxPolyDP(biggest_contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            points = np.float32(approx.reshape(4, 2))
            rect = np.zeros((4, 2), dtype="float32")
            
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]
            rect[2] = points[np.argmax(s)]
            
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]
            rect[3] = points[np.argmax(diff)]
            
            pts2 = np.float32([[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]])
            matrix = cv2.getPerspectiveTransform(rect, pts2)
            warped_img = cv2.warpPerspective(img, matrix, (img_width, img_height))
            warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            
            # ধাপ ৪: বৃত্তগুলো চিহ্নিত করা এবং উত্তর বের করা
            # এই অংশটি আপনার OMR শিটের ডিজাইনের উপর নির্ভরশীল। এখানে একটি সাধারণ ডিজাইনের জন্য কোড লেখা হলো।
            rows = np.vsplit(warped_gray, TOTAL_QUESTIONS)
            user_answers = {}

            for r_idx, r in enumerate(rows):
                cols = np.hsplit(r, OPTIONS_COUNT)
                pixel_values = []
                for c in cols:
                    total_pixels = cv2.countNonZero(c)
                    pixel_values.append(total_pixels)
                
                # সবচেয়ে কম সাদা পিক্সেল মানে সবচেয়ে বেশি কালো, অর্থাৎ ভরাট করা বৃত্ত
                min_val = min(pixel_values)
                # একটি থ্রেশহোল্ড সেট করা, যাতে প্রায় খালি বৃত্তকে উত্তর হিসেবে ধরা না হয়
                if min_val < 3500: # এই মানটি আপনার ছবির রেজোলিউশন ও কালির গাঢ়ত্বের উপর নির্ভর করবে
                    user_answers[r_idx] = pixel_values.index(min_val)

            # ধাপ ৫: স্কোর গণনা করা
            correct = 0
            wrong = 0
            for q_num, user_ans in user_answers.items():
                if q_num in ANSWER_KEY and user_ans == ANSWER_KEY[q_num]:
                    correct += 1
                else:
                    wrong += 1
            
            unanswered = TOTAL_QUESTIONS - len(user_answers)
            score = correct

            result_data = {
                'score': score,
                'correct': correct,
                'wrong': wrong,
                'unanswered': unanswered,
                'totalQuestions': TOTAL_QUESTIONS,
                'processedAt': firestore.SERVER_TIMESTAMP
            }

        else:
            # যদি OMR শিট ঠিকমতো ডিটেক্ট না হয়
            result_data = {'error': 'Could not detect the OMR sheet properly. Please upload a clearer image.'}

        # ফলাফল Firestore-এ সেভ করা
        db = firestore.client()
        db.collection('users').document(user_id).collection('results').document(upload_id).set(result_data)
        print(f"Result saved to Firestore for user {user_id}, upload {upload_id}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # কোনো সমস্যা হলে Firestore-এ এরর মেসেজ পাঠানো
        db = firestore.client()
        db.collection('users').document(user_id).collection('results').document(upload_id).set({
            'error': str(e),
            'processedAt': firestore.SERVER_TIMESTAMP
        })
    finally:
        # টেম্পোরারি ফাইল ডিলিট করা
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file {temp_file_path} deleted.")


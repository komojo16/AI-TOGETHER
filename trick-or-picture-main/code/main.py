import socketio
import cv2
import mediapipe as mp
import numpy as np
import math
import picture
import base64
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import os
import temp

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 소켓 클라이언트 초기화 및 재연결 설정
sio = socketio.AsyncClient(
    reconnection=True,
    reconnection_attempts=10,
    reconnection_delay=1,
    reconnection_delay_max=5
)

# Mediapipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

# 쓰레드 풀
executor = ThreadPoolExecutor(max_workers=4)  # 이미지 처리를 위한 스레드 풀


# 얼굴 각도 계산 함수
def calculate_face_angle(face_landmarks, image_width, image_height):
    try:
        left_eye = [
            (face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2,
            (face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2
        ]
        right_eye = [
            (face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2,
            (face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2
        ]

        dx = (right_eye[0] - left_eye[0]) * image_width
        dy = (right_eye[1] - left_eye[1]) * image_height
        eye_angle = math.degrees(math.atan2(dy, dx))
        return max(-45, min(45, eye_angle))
    except Exception as e:
        logger.error(f"Error calculating face angle: {e}")
        return 0


# 얼굴 메시 처리 및 필터 적용 함수
def apply_face_mesh_sync(image, face_mesh, filter_image_path):
    image_height, image_width, _ = image.shape
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            angle = calculate_face_angle(face_landmarks, image_width, image_height)
            x = int(face_landmarks.landmark[1].x * image_width)
            y = int(face_landmarks.landmark[1].y * image_height)
            x1 = face_landmarks.landmark[152].x * image_width
            y1 = face_landmarks.landmark[152].y * image_height
            x2 = face_landmarks.landmark[10].x * image_width
            y2 = face_landmarks.landmark[10].y * image_height
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            filter_height = int(distance)
            filter_width = int(filter_height * 0.5)

            image = picture.take_pictures_start(
                filter_image_path, image, x, y,
                filter_width * 2, filter_height * 2,
                int(angle)
            )
    return image



async def apply_face_mesh_async(image, face_mesh, filter_image_path):
    loop = asyncio.get_running_loop()
    # 이미지는 비동기적으로 처리하기 위해 run_in_executor 사용
    # 이 함수는 동기적인 작업인 apply_face_mesh_sync를 비동기적으로 실행
    return await loop.run_in_executor(
        executor,  # executor는 별도의 스레드에서 작업을 수행하게 해줌
        apply_face_mesh_sync,  # 동기 함수
        image,
        face_mesh,
        filter_image_path
    )


# 소켓 이벤트 핸들러
@sio.event
async def connect():
    print(12)
    logger.info("Connected to the server.")
    await sio.emit('register', {'role': 'ai'})


@sio.event
async def disconnect():
    logger.warning("Disconnected from the server.")

'''
@sio.on('end')
async def in_image(data):
    end_frame = base64.b64decode(data['end_frame'] + '==')
    end_img1 = base64.b64decode(data['end_img1'] + '==')
    end_img2 = base64.b64decode(data['end_img2'] + '==')
    temp.img_connect(end_frame, end_img1, end_img2)
'''

@sio.on('input')
async def receive_image(data):
    print(3333)
    try:
        img_data = base64.b64decode(data['image'] + '==')
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        filter_number = data.get('filter_number', 0)
        logger.info(f"Received filter number: {filter_number}")

        FILTER_DIRECTORY = r'C:\Users\kyle0\Desktop\trick-or-picture-main\trick-or-picture-main\img'  # 필터 이미지 디렉토리

        filter_image = os.path.join(FILTER_DIRECTORY, f"{filter_number}.png")  # 경로 안전하게 결합
        print(f"Trying to find filter image at: {filter_image}")

        if os.path.isfile(filter_image):
            print("OK")
            filter_image_path = filter_image
        else:
            print(f"File does not exist: {filter_image}")
            logger.error("Failed to decode image")

        # BGR에서 RGB로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = True

        # 비동기로 이미지 처리
        processed_image = await apply_face_mesh_async(image, face_mesh, filter_image_path)

        # RGB에서 BGR로 변환
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        # 필터링된 이미지를 Base64로 인코딩
        _, buffer = cv2.imencode('.jpg', processed_image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        await sio.emit('output', jpg_as_text)
        logger.info("Processed image sent successfully.")

    except Exception as e:
        logger.error(f"Error during image processing: {e}")

async def main():
    try:
        #await sio.connect('http://121.159.74.206:8888', transports=['websocket'])
        await sio.connect('http://121.159.74.206:8888', transports=['websocket'])
        await sio.wait()  # 서버의 이벤트를 대기
    except Exception as e:
        logger.error(f"Socket connection error: {e}")
    finally:
        face_mesh.close()
        executor.shutdown(wait=True)
        logger.info("Resources have been cleaned up.")


# 서버와 연결 및 대기
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


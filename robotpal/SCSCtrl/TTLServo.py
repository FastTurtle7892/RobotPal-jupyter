import time
import numpy as np
from .._core.server import SimulatorServer

# --- [설정] JetTank 하드웨어 규격 (수정 금지) ---
linkageLenA = 90
linkageLenB = 160

# 서보 설정 (원본 유지)
servoNumCtrl = [0, 1]
servoDirection = [1, -1] 

servoInputRange = 850
servoAngleRange = 180
servoInit = [None, 512, 512, 512, 512, 512] 

_server = SimulatorServer.instance()

# ---------------------------------------------------------
# [Sim] 시뮬레이터 통신용 함수 (가상 드라이버)
# ---------------------------------------------------------
def syncCtrl(ID_List, Speed_List, Goal_List):
    """
    원본 코드의 syncCtrl을 시뮬레이터용으로 대체
    Goal_List에는 0~1023 사이의 Raw Step 값이 들어옵니다.
    """
    for i in range(len(ID_List)):
        servo_id = ID_List[i]
        raw_goal = Goal_List[i]
        raw_speed = Speed_List[i]

        # 1. Raw Step (0~1023) -> Degree 변환
        # JetTank 기준: 512=0도, 1023=약+100도
        degree = (raw_goal - 512) * (servoAngleRange / servoInputRange)

        # 2. [중요] ID 5번(카메라 상하) 방향 보정
        # 실제 로봇과 시뮬레이터의 모터 장착 방향 차이 보정
        if servo_id == 5:
            degree = -degree 

        # 3. 속도 스케일링
        sim_speed = float(raw_speed) * 0.2
        if sim_speed <= 0: sim_speed = 0

        # 4. 서버 전송
        _server.update_servo_value(servo_id, degree, sim_speed)

def servoAngleCtrl(ServoNum, AngleInput, DirectionDebug, SpeedInput):
    """원본 함수: 각도를 받아 오프셋 계산 후 전송"""
    offsetGenOut = servoInit[ServoNum] + int((servoInputRange/servoAngleRange)*AngleInput*DirectionDebug)
    
    # 계산된 Raw 값을 syncCtrl로 보냄
    syncCtrl([ServoNum], [SpeedInput], [offsetGenOut])
    return offsetGenOut

# ---------------------------------------------------------
# [Real Logic] JetTank 원본 기하학 공식 (건드리지 않음)
# ---------------------------------------------------------
def limitCheck(posInput, circlePos, circleLen, outline):
    """두 원의 교차점 및 작업 영역 제한 계산"""
    circleRx = posInput[0]-circlePos[0]
    circleRy = posInput[1]-circlePos[1]
    realPosSquare = circleRx*circleRx+circleRy*circleRy
    shortRadiusSquare = np.square(circleLen[1]-circleLen[0])
    longRadiusSquare = np.square(circleLen[1]+circleLen[0])

    if realPosSquare >= shortRadiusSquare and realPosSquare <= longRadiusSquare:
        return posInput[0], posInput[1]
    else:
        lineK = (posInput[1]-circlePos[1])/(posInput[0]-circlePos[0]) if (posInput[0]-circlePos[0]) != 0 else 0
        lineB = circlePos[1]-(lineK*circlePos[0])
        
        if realPosSquare < shortRadiusSquare:
            aX = 1 + lineK*lineK
            bX = 2*lineK*(lineB - circlePos[1]) - 2*circlePos[0]
            cX = circlePos[0]*circlePos[0] + (lineB - circlePos[1])*(lineB - circlePos[1]) - shortRadiusSquare
            resultX = bX*bX - 4*aX*cX
            if resultX < 0: resultX = 0 
            x1 = (-bX + np.sqrt(resultX))/(2*aX)
            x2 = (-bX - np.sqrt(resultX))/(2*aX)
            y1 = lineK*x1 + lineB
            y2 = lineK*x2 + lineB
            
            # (복잡한 원본 분기 로직 간소화하여 가장 가까운 점 선택)
            # 시뮬레이션 안정성을 위해 도달 가능한 최단 거리 지점을 반환
            dist1 = (posInput[0]-x1)**2 + (posInput[1]-y1)**2
            dist2 = (posInput[0]-x2)**2 + (posInput[1]-y2)**2
            if dist1 < dist2: return x1, y1
            else: return x2, y2

        elif realPosSquare > longRadiusSquare:
            # 최대 거리 제한 로직
            unit_vec = np.array([circleRx, circleRy])
            norm = np.linalg.norm(unit_vec)
            if norm == 0: return posInput[0], posInput[1]
            scaled = unit_vec / norm * (circleLen[1]+circleLen[0] - outline)
            return circlePos[0] + scaled[0], circlePos[1] + scaled[1]
            
    return posInput[0], posInput[1]

def planeLinkageReverse(linkageLen, linkageEnDe, servoNum, debugPos, goalPos):
    """JetTank IK 핵심 함수"""
    # 1. 좌표 보정
    goalPos[0] = goalPos[0] + debugPos[0]
    goalPos[1] = goalPos[1] + debugPos[1]

    AngleEnD = np.arctan(linkageEnDe/linkageLen[1])*180/np.pi
    linkageLenREAL = np.sqrt(((linkageLen[1]*linkageLen[1])+(linkageEnDe*linkageEnDe)))

    # 2. Limit Check
    goalPos[0], goalPos[1] = limitCheck(goalPos, debugPos, [linkageLen[0], linkageLenREAL], 0.00001)

    # 3. IK Calculation (원본 로직)
    # X >= 0 (앞쪽)인 경우만 주로 사용
    if goalPos[0] >= 0:
        sqrtGenOut = np.sqrt(goalPos[0]*goalPos[0]+goalPos[1]*goalPos[1])
        if sqrtGenOut == 0: return [0, 0]

        nGenOut = (linkageLen[0]*linkageLen[0]+goalPos[0]*goalPos[0]+goalPos[1]*goalPos[1]-linkageLenREAL*linkageLenREAL)/(2*linkageLen[0]*sqrtGenOut)
        
        # acos 범위 보호
        nGenOut = max(-1, min(1, nGenOut))
        angleA = np.arccos(nGenOut)*180/np.pi

        AB = goalPos[1]/goalPos[0] if goalPos[0] != 0 else 99999
        angleB = np.arctan(AB)*180/np.pi
        
        angleGenA = angleB - angleA

        mGenOut = (linkageLen[0]*linkageLen[0]+linkageLenREAL*linkageLenREAL-goalPos[0]*goalPos[0]-goalPos[1]*goalPos[1])/(2*linkageLen[0]*linkageLenREAL)
        mGenOut = max(-1, min(1, mGenOut))
        angleGenB = np.arccos(mGenOut)*180/np.pi - 90

        linkagePointC = 0
        anglePosC = angleGenB + angleGenA

        # 결과 반환: [AngleA, AngleB, ...]
        return [angleGenA*servoDirection[servoNumCtrl[0]], (angleGenB+AngleEnD)*servoDirection[servoNumCtrl[1]], 0, linkagePointC, anglePosC]
    
    # X < 0 인 경우는 시뮬레이션에서 0으로 처리 (안정성)
    return [0, 0, 0, 0, 0]

# ---------------------------------------------------------
# 3. 사용자 호출 함수 (xyInput)
# ---------------------------------------------------------
def xyInput(xInput, yInput):
    # 원본 파일에 있는 그대로 사용
    angGenOut = planeLinkageReverse([linkageLenA, linkageLenB], 0, servoNumCtrl, [0,0], [xInput, -yInput])
    
    # [중요] 원본 코드의 오프셋 적용 (+90)
    # JetTank는 2번 모터에 +90도 오프셋을 줍니다.
    servoAngleCtrl(2, angGenOut[0] + 90, 1, 200) # ID 2
    servoAngleCtrl(3, angGenOut[1], -1, 200)     # ID 3

    return [angGenOut[0], angGenOut[1]]

# 호환성 함수들
def servoStop(n): pass
def portClose(): pass
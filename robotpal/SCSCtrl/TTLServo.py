import time
import numpy as np
from .._core.server import SimulatorServer

# --- [설정] JetTank 하드웨어 규격 (Real 코드 유지) ---
linkageLenA = 90
linkageLenB = 160

servoNumCtrl = [0, 1]
servoDirection = [1, -1] # [ID2, ID3] 방향

servoInputRange = 850
servoAngleRange = 180

# 초기 위치 (Center: 512)
servoInit = [None, 512, 512, 512, 512, 512]

# [Sim] 내부 상태 추적용
nowPos = [None, 512, 512, 512, 512, 512] 
nextPos = [None, 512, 512, 512, 512, 512]
speedBuffer = [None, 512, 512, 512, 512, 512]

_server = SimulatorServer.instance()

# ---------------------------------------------------------
# [Sim] 시뮬레이터 통신 및 좌표계 보정 (핵심 수정)
# ---------------------------------------------------------
# ---------------------------------------------------------
# [Sim] 시뮬레이터 통신 및 좌표계 보정 (Zero Point Calibration)
# ---------------------------------------------------------
def syncCtrl(ID_List, Speed_List, Goal_List):
    """
    실제 로봇의 Raw Step 값을 받아, 시뮬레이터가 사진1(초기자세)과 같아지도록 보정하여 전송
    """
    for i in range(len(ID_List)):
        servo_id = ID_List[i]
        raw_goal = Goal_List[i]
        raw_speed = Speed_List[i]

        degree = (raw_goal - 512) * (servoAngleRange / servoInputRange)

        # [ID 2: Shoulder]
        # xyInput에서 +90을 뺐으므로, 여기서는 있는 그대로(degree) 받으면 
        # 수직(0)과 숙임(-각도)이 정상 작동합니다.
        if servo_id == 2:
            degree = -degree

        # [ID 3: Elbow]
        # 초기화 시 'ㄴ'자로 굽히기 위한 보정 (-90) 유지
        elif servo_id == 3:
            degree = degree

        # 나머지 ID 유지
        elif servo_id == 4:
            degree = degree 
        elif servo_id == 5:
            degree = degree
        elif servo_id == 1:
            pass

        sim_speed = float(raw_speed) * 0.2
        if sim_speed <= 0.1: sim_speed = 10.0
        
        _server.update_servo_value(servo_id, degree, sim_speed)
        nowPos[servo_id] = raw_goal

def servoAngleCtrl(ServoNum, AngleInput, DirectionDebug, SpeedInput):
    offsetGenOut = servoInit[ServoNum] + int((servoInputRange/servoAngleRange)*AngleInput*DirectionDebug)
    syncCtrl([ServoNum], [SpeedInput], [offsetGenOut])
    return offsetGenOut

def returnOffset(ServoNum, AngleInput, DirectionDebug):
    return servoInit[ServoNum] + int((servoInputRange/servoAngleRange)*AngleInput*DirectionDebug)

# ---------------------------------------------------------
# [Math] IK 연산 (Real 코드 동일)
# ---------------------------------------------------------
def limitCheck(posInput, circlePos, circleLen, outline):
    circleRx = posInput[0]-circlePos[0]
    circleRy = posInput[1]-circlePos[1]
    realPosSquare = circleRx*circleRx+circleRy*circleRy
    shortRadiusSquare = np.square(circleLen[1]-circleLen[0])
    longRadiusSquare = np.square(circleLen[1]+circleLen[0])

    if realPosSquare >= shortRadiusSquare and realPosSquare <= longRadiusSquare:
        return posInput[0], posInput[1]
    else:
        denom = (posInput[0]-circlePos[0])
        lineK = (posInput[1]-circlePos[1])/denom if denom != 0 else 0
        lineB = circlePos[1]-(lineK*circlePos[0])
        
        if realPosSquare < shortRadiusSquare:
            aX = 1 + lineK*lineK
            bX = 2*lineK*(lineB - circlePos[1]) - 2*circlePos[0]
            cX = circlePos[0]*circlePos[0] + (lineB - circlePos[1])*(lineB - circlePos[1]) - shortRadiusSquare
            resultX = max(0, bX*bX - 4*aX*cX)
            
            x1 = (-bX + np.sqrt(resultX))/(2*aX)
            x2 = (-bX - np.sqrt(resultX))/(2*aX)
            y1 = lineK*x1 + lineB
            y2 = lineK*x2 + lineB

            if ((posInput[0]-x1)**2 + (posInput[1]-y1)**2) < ((posInput[0]-x2)**2 + (posInput[1]-y2)**2):
                return x1, y1
            else: return x2, y2

        elif realPosSquare > longRadiusSquare:
            vec_x, vec_y = circleRx, circleRy
            mag = np.sqrt(vec_x**2 + vec_y**2)
            if mag == 0: return posInput[0], posInput[1]
            limit_radius = circleLen[1] + circleLen[0] - outline
            return circlePos[0] + (vec_x/mag)*limit_radius, circlePos[1] + (vec_y/mag)*limit_radius
            
    return posInput[0], posInput[1]

def planeLinkageReverse(linkageLen, linkageEnDe, servoNum, debugPos, goalPos):
    gp = [goalPos[0] + debugPos[0], goalPos[1] + debugPos[1]]
    AngleEnD = np.arctan(linkageEnDe/linkageLen[1])*180/np.pi
    linkageLenREAL = np.sqrt(((linkageLen[1]*linkageLen[1])+(linkageEnDe*linkageEnDe)))
    gp[0], gp[1] = limitCheck(gp, debugPos, [linkageLen[0], linkageLenREAL], 0.00001)

    if gp[0] < 0:
        gp[0] = -gp[0]
        mGenOut = linkageLenREAL**2 - linkageLen[0]**2 - gp[0]**2 - gp[1]**2
        nGenOut = mGenOut/(2*linkageLen[0])
        val = max(-1, min(1, nGenOut/np.sqrt(gp[0]**2+gp[1]**2)))
        angleGenA = np.arctan(gp[1]/gp[0]) + np.arcsin(val)
        val2 = max(-1, min(1, (gp[1]-linkageLen[0]*np.cos(angleGenA))/linkageLenREAL))
        angleGenB = np.arcsin(val2)-angleGenA
        angleGenA = 90 - angleGenA*180/np.pi
        angleGenB = angleGenB*180/np.pi
        return [angleGenA*servoDirection[servoNumCtrl[0]], (angleGenB+AngleEnD)*servoDirection[servoNumCtrl[1]]]

    elif gp[0] == 0:
        if gp[1] == 0: gp[1] = 0.0001
        val = max(-1, min(1, (linkageLen[0]**2+gp[1]**2-linkageLenREAL**2)/(2*linkageLen[0]*gp[1])))
        angleGenA = np.arccos(val)
        cGenOut = np.tan(angleGenA)*linkageLen[0]
        dGenOut = gp[1]-(linkageLen[0]/np.cos(angleGenA)) if np.cos(angleGenA) != 0 else 0
        val2 = max(-1, min(1, (cGenOut**2+linkageLenREAL**2-dGenOut**2)/(2*cGenOut*linkageLenREAL)))
        angleGenB = np.arccos(val2)
        return [(-angleGenA*180/np.pi + 90)*servoDirection[servoNumCtrl[0]], (-angleGenB*180/np.pi+AngleEnD)*servoDirection[servoNumCtrl[1]]]

    elif gp[0] > 0:
        sqrtGenOut = np.sqrt(gp[0]**2+gp[1]**2)
        val = max(-1, min(1, (linkageLen[0]**2+gp[0]**2+gp[1]**2-linkageLenREAL**2)/(2*linkageLen[0]*sqrtGenOut)))
        angleA = np.arccos(val)*180/np.pi
        angleB = np.arctan(gp[1]/gp[0])*180/np.pi
        angleGenA = angleB - angleA
        val2 = max(-1, min(1, (linkageLen[0]**2+linkageLenREAL**2-gp[0]**2-gp[1]**2)/(2*linkageLen[0]*linkageLenREAL)))
        angleGenB = np.arccos(val2)*180/np.pi - 90
        return [angleGenA*servoDirection[servoNumCtrl[0]], (angleGenB+AngleEnD)*servoDirection[servoNumCtrl[1]]]
    
    return [0,0]

def speedGenOut(servoNum, dTime):
    dPos = abs(nextPos[servoNum] - nowPos[servoNum])
    return int(round(dPos/dTime, 0)) if dTime > 0 else 0

def xyInput(xInput, yInput):
    # [수정 1] -yInput -> yInput (시뮬레이터 좌표계 방향 맞춤)
    angGenOut = planeLinkageReverse([linkageLenA, linkageLenB], 0, servoNumCtrl, [0,0], [xInput, yInput])
    
    # [수정 2] ID 2에서 '+90' 제거
    # 시뮬레이터는 0도가 이미 수직이므로, 하드웨어용 보정값(+90)이 필요 없습니다.
    servoAngleCtrl(2, angGenOut[0], -1, 300) 
    
    # ID 3은 그대로 유지
    servoAngleCtrl(3, angGenOut[1], -1, 300)
    
    return [angGenOut[0], angGenOut[1]]

def xyInputSmooth(xInput, yInput, dt):
    # [수정 1] -yInput -> yInput
    angGenOut = planeLinkageReverse([linkageLenA, linkageLenB], 0, servoNumCtrl, [0,0], [xInput, yInput])
    
    # [수정 2] ID 2에서 '+90' 제거
    nextPos[2] = returnOffset(2, angGenOut[0], 1) # 여기도 +90 제거
    nextPos[3] = returnOffset(3, angGenOut[1], -1)
    
    spd2, spd3 = speedGenOut(2, dt), speedGenOut(3, dt)
    
    # [수정 2] ID 2에서 '+90' 제거
    servoAngleCtrl(2, angGenOut[0], 1, spd2)
    servoAngleCtrl(3, angGenOut[1], -1, spd3)
    
    return [angGenOut[0], angGenOut[1]]

# 호환성 더미
def servoStop(n): pass
def portClose(): pass
def stopServo(n): pass
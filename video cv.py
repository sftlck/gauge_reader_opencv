import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Set
import matplotlib.pyplot as plt
from enum import Enum

###### ESCREVI ESSE CÓDIGO AQUI OUVINDO "BUNDY" - ANIMAL ALPHA

class CalibrationStage(Enum):
    ###### CASTRO 25/02/2026 - CLASSE DE ESTADOS PARA SELEÇÃO DE PONTOS MÍNIMO E MÁXIMO
    NOT_CALIBRATING =       0
    SELECTING_MIN =         1
    SELECTING_MAX =         2

# =======================
# CLASSE PARA GLOBALIZAR OS DADOS DO INDICADOR
# ============================================================================

@dataclass
class CameraParams:
    
    fx:                     float =                     426.7231198260606
    fy:                     float =                     424.44185129184774
    skew:                   float =                     0.0
    cx:                     float =                     308.3359965293331
    cy:                     float =                     233.71108395163847
    width:                  int =                       640
    height:                 int =                       480
    r0:                     int =                       2316661
    r1:                     int =                       -32512066

@dataclass
class ProcessingParams:
    n_regions:              int =                       40
    min_region_area:        int =                       10
    canny1:                 int =                       10
    canny2:                 int =                       255
    thresh_value:           int =                       214
    thresh_value2:          int =                       255
    dp:                     int =                       7
    param1:                 int =                       102
    param2:                 int =                       306 
    min_radius:             int =                       90
    max_radius:             int =                       100
    distance:               int =                       201
    history:                int =                       300
    var_threshold:          int =                       120
    gamma:                  float =                     1.0
    alpha:                  float =                     1.5
    beta:                   float =                     0.0
    min_contour_area:       int =                       15
    ring_scale_factor:      float =                     0.8
    ring_thresh_low:        int =                       170
    ring_thresh_high:       int =                       255
    value_range:            float =                     35.0
    resolution:             float =                     0.1
    num_bins:               int =                       360
    threshold_value:        int =                       127
    limit_chart:            int =                       25
    extension_beyond:       int =                       15  

@dataclass                      
class ROIConfig:                        
    center_x:               int =                       326
    center_y:               int =                       267
    radius:                 int =                       118
    selected:               bool =                      True

@dataclass
class LineData:
    ###### CASTRO 25/02/2026 - ESSA CLASSE SERVE SÓ PARA GUARDAR TODAS AS PROPRIEDADES INTERESSANTES DE CADA LINHA NA TELA
    centroid:               Tuple[int, int]  
    center:                 Tuple[int, int]    
    angle_from_min:         float =                     0.0  
    absolute_angle:         float =                     0.0  
    circle_intersection:    Optional[Tuple[int, int]] = None  
    extended_end:           Optional[Tuple[int, int]] = None  
    is_tick_mark:           bool =                      False
    index:                  int =                       -1
    ###### CASTRO 25/02/2026 - ADICIONEI A RESOLUÇÃO INDEXADA A ESSA LINHA
    res:                    int =                       -1

@dataclass
class NeedleData:
    ###### CASTRO 25/02/2026 - ESTA CLASSE SEGURA AS INFORMAÇÕES SOBRE O PONTEIRO DETECTADO
    centroid:               Optional[Tuple[int, int]] = None  
    tip:                    Optional[Tuple[int, int]] = None  
    circle_intersection:    Optional[Tuple[int, int]] = None  
    absolute_angle:         float =                     0.0  
    relative_angle:         float =                     0.0  
    
    def as_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if self.tip:
            return (self.tip, self.centroid)  
        return None

@dataclass
class GaugeData:
    ###### CASTRO 25/02/2026 - ESTA CLASSE SEGURA AS PRINCIPAIS INFORMAÇÕES DO SISTEMA PARA REPASSAR A ALGUM SISTEMA DE CONTROLE EXTERNO
    center:                 Tuple[int, int] =       (0, 0)
    radius:                 int =                   0
    circle_detected:        bool =                  False
    
    min_line:               Optional[LineData] =    None
    max_line:               Optional[LineData] =    None
    is_calibrated:          bool =                  False
    
    ###### CASTRO 25/02/2026 - AGORA AS LINHAS SÃO PERMANENTES APÓS CALIBRAÇÃO
    permanent_lines:        List[LineData] =        field(default_factory=list)
    tick_mark_angles:       List[float] =           field(default_factory=list)
    scale_values:           List[int] =             field(default_factory=list) 

    needle:                 NeedleData =            field(default_factory=NeedleData)

    crossed_angles:         Dict[float, int] =      field(default_factory=dict)
    max_crossed:            bool =                  False    

    ###### CASTRO 25/02/2026 - PARÂMETROS DO INDICADOR
    min_value:              float =                 2
    max_value:              float =                 60
    res:                    int =                   2
    unit:                   str =                   "mca"

    total_angle_span:       float =                 0.0
    current_read:           float =                 0.0

    ###### CASTRO 25/02/2026 - ADICIONEI ALERTAS DE MÁXIMO COMO FLAG DE ESTADO-TEMPORAL E ESTADO-INSTANTÂNEO
    warning_max_flag:       bool =                  False
    warning_max_instant:    bool =                  False
    
    ###### CASTRO 25/02/2026 - FLAG PARA INDICAR QUE AS LINHAS JÁ FORAM CAPTURADAS
    lines_captured:         bool =                  False
    
    def update_tick_marks(self):
        
        ###### CASTRO 25/02/2026 - AQUI ELE ORDENA O ÂNGULO DA LINHA EXISTENTE ENTRE ELA MESMA E A LINHA DE MÍNIMO A MENOS DA LINHA DE ORIGEM
        self.tick_mark_angles =     sorted([line.angle_from_min for line in self.permanent_lines if line.is_tick_mark])
        self.scale_values =         sorted([line.res for line in self.permanent_lines if line.is_tick_mark])
        
    def get_crossed_count(self) -> int:
        return len(self.crossed_angles)
    
    def get_progress(self) -> Tuple[int, int]:
        return (len(self.crossed_angles), len(self.tick_mark_angles))
    
    def reset_tracking(self):
        self.crossed_angles.clear()
        self.max_crossed = False
    
    def get_max_relative_angle(self) -> float:
        if not self.min_line or not self.max_line:
            return 0.0
        return self.max_line.angle_from_min
    
    ##### CASTRO 25/02/2026 - DECIDI QUE EM VEZ DE DESENHAR TUDO A TODO MOMENTO NOS FRAMES, VOU DESENHAR SÓ EM UM FRAME E TORNAR AS LINHAS PERMANENTES
    def reset_calibration(self):
        self.min_line = None
        self.max_line = None
        self.is_calibrated = False
        self.permanent_lines.clear()
        self.lines_captured = False
        self.tick_mark_angles.clear()
        self.scale_values.clear()
        self.crossed_angles.clear()
        self.max_crossed = False
        self.warning_max_flag = False
        self.warning_max_instant = False
        self.current_read = 0.0
        self.total_angle_span = 0.0

# ========================
# ESTADO DE PROCESSAMENTO
# ============================================================================

class ProcessingState:
    
    def __init__(self):
        self.frame_counter:         int = 0
        self.calibration_mode:      bool = False
        self.calibration_stage:     CalibrationStage = CalibrationStage.NOT_CALIBRATING
        self.frozen_frame:          Optional[np.ndarray] = None

# =========================================
# CAMERA CALIBRATOR (UNCHANGED)
# ============================================================================

class CameraCalibrator:
    
    def __init__(self, params: CameraParams):
        self.params = params
        self.camera_matrix = self._create_camera_matrix()
        self.dist_coeffs = self._create_distortion_coefficients()
        self.map1, self.map2, self.roi, self.new_camera_matrix = self._setup_undistort_maps()
    
    def _create_camera_matrix(self) -> np.ndarray:
        return np.array([
            [self.params.fx, self.params.skew, self.params.cx],
            [0, self.params.fy, self.params.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _create_distortion_coefficients(self) -> np.ndarray:

        scale_factor = 1e9
        k1 = self.params.r0 / scale_factor
        k2 = self.params.r1 / scale_factor
        return np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    def _setup_undistort_maps(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray]:

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix( self.camera_matrix, self.dist_coeffs,  (self.params.width, self.params.height), 1,  (self.params.width, self.params.height) )
        map1, map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, new_camera_matrix,(self.params.width, self.params.height), cv2.CV_16SC2)
        return map1, map2, roi, new_camera_matrix
    
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        
        undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LANCZOS4)
        x, y, w, h = self.roi
        return undistorted[y:y+h, x:x+w]

# ================
# ANGLE CALCULATOR
# ============================================================================

class AngleCalculator:
    
    def __init__(self, value_range: float, resolution: float):
        self.value_range = value_range
        self.resolution = resolution
        self.res_calc = self._calculate_resolution_precision(resolution)
    
    @staticmethod
    def _calculate_resolution_precision(resolution: float) -> int:
        res_str = str(resolution)
        return res_str.find("1") - 1
    
    @staticmethod
    def calculate_angle_from_vector(dx: float, dy: float) -> float:

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
         
        if angle_deg < 0:
            angle_deg += 360
        elif angle_deg == 0:
            angle_deg = 0.000001
        return angle_deg
    
    def calculate_line_angle(self, line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:

        point, center = line
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return self.calculate_angle_from_vector(dx, dy)
    
    @staticmethod
    def directed_angle(line1: Tuple[Tuple[int, int], Tuple[int, int]], line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        
        point1, center1 = line1
        point2, center2 = line2
        
        v1,v2 = np.array([point1[0] - center1[0], point1[1] - center1[1]], dtype=np.float64), np.array([point2[0] - center2[0], point2[1] - center2[1]], dtype=np.float64)
        
        v1_norm, v2_norm = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_magnitude = np.degrees(np.arccos(dot))
        
        cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
        
        if cross < 0:
            return float(angle_magnitude)
        else:
            return float(360 - angle_magnitude)
    
    def relative_angle_from_min(self, line: Tuple[Tuple[int, int], Tuple[int, int]], min_line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:

        return self.directed_angle(min_line, line)
    
    def angle_to_value(self, angle: float, max_angle: float) -> float:

        if max_angle == 0:
            return 0
        return round(self.value_range * (angle / max_angle), self.res_calc)
    
    @staticmethod
    def find_line_circle_intersections(line_start: Tuple[int, int],  line_end: Tuple[int, int],  circle_center: Tuple[int, int],  circle_radius: int) -> List[Tuple[int, int]]:
        x1, y1 = line_start
        x2, y2 = line_end
        cx, cy = circle_center
        r = circle_radius
        
        dx, dy = x2 - x1,  y2 - y1

        fx, fy = x1 - cx, y1 - cy
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - r*r
        
        discriminant = b*b - 4*a*c
        intersections = []
        
        if discriminant >= 0 and a != 0:
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            
            for t in [t1, t2]:
                if 0 <= t <= 1:
                    ix = int(x1 + t * dx)
                    iy = int(y1 + t * dy)
                    intersections.append((ix, iy))
        
        return intersections

# =========================+
# IMAGE PROCESSING UTILITIES
# ============================================================================

class ImageProcessor:
    
    def __init__(self, params: ProcessingParams, app: 'GaugeReaderApp' = None):
        self.params = params
        self.app = app
        self.angle_calculator = AngleCalculator(params.value_range, params.resolution)
        
        self.last_component_centroids = []
        self.last_offset = None
    
    def analyze_ring(self, frame: np.ndarray, offset: Optional[Tuple[int, int]], gauge_data: GaugeData, result_frame: np.ndarray, original_frame: np.ndarray) -> List[LineData]:

        img_copy = frame.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, self.params.threshold_value, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
        
        detected_lines = []
        component_centroids = []
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            
            ###### CASTRO 25/02/2026 - FIXEI A ÁREA EM 5 MAS TALVEZ TENHA DE REDUZIR PARA INDICADORES COM TRAÇOS MAIS FINOS
            if area:
                x,y,w,h = stats[label, cv2.CC_STAT_LEFT],  stats[label, cv2.CC_STAT_TOP],stats[label, cv2.CC_STAT_WIDTH],stats[label, cv2.CC_STAT_HEIGHT]

                centroid = (int(centroids[label][0]), int(centroids[label][1]))
                component_centroids.append(centroid)
                
                if offset:
                    original_centroid = (offset[0] + centroid[0], offset[1] + centroid[1])
                else:
                    original_centroid = centroid

                line = LineData( centroid=original_centroid, center=gauge_data.center, absolute_angle=self.angle_calculator.calculate_angle_from_vector( original_centroid[0] - gauge_data.center[0], original_centroid[1] - gauge_data.center[1] ) )

                self._calculate_line_intersection(line, gauge_data)
                
                detected_lines.append(line)
        
        self.last_component_centroids = component_centroids
        self.last_offset = offset
        
        self._draw_debug_regions(img_copy, component_centroids, stats)
        
        return detected_lines
    
    def _calculate_line_intersection(self, line: LineData, gauge_data: GaugeData):

        dx, dy = line.centroid[0] - line.center[0], line.centroid[1] - line.center[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > 0:
            dx_norm, dy_norm = dx / dist, dy / dist
            circle_x, circle_y = int(line.center[0] + dx_norm * gauge_data.radius), int(line.center[1] + dy_norm * gauge_data.radius)
            line.circle_intersection = (circle_x, circle_y)
            
            line.extended_end = ( int(circle_x + dx_norm * self.params.extension_beyond), int(circle_y + dy_norm * self.params.extension_beyond) )
            line.extended_end_text = ( int(circle_x + dx_norm * self.params.extension_beyond + self.params.extension_beyond * .2), int(circle_y + dy_norm * self.params.extension_beyond + self.params.extension_beyond * .2) )
    
    def _draw_debug_regions(self, img: np.ndarray, centroids: List[Tuple[int, int]], stats: np.ndarray):
        if len(centroids) > 0:
            display_img = img.copy()
            ###### CASTRO 25/02/2026 - AQUI EU TAQUEI COMO CONSTANTE AS 50 REGIÕES DETECTADAS, MAS SEI LÁ, TALVEZ SEJA INSUFICIENTE. NUM FUTURO PRÓXIMO EU CORRIJO PARA TODAS A REGIÕES DETECATADAS
            for i, region in enumerate(stats[0:50]):
                x, y, w, h = region[cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT+1]
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            #cv2.imshow('detected_regions', display_img)
    
    def draw_lines_on_frame(self, result_frame: np.ndarray, original_frame: np.ndarray, lines: List[LineData], gauge_data: GaugeData):
        ###### CASTRO 25/02/2026 - ESTA FUNÇÃO AGORA DESENHA AS LINHAS PERMANENTES
        for i, line in enumerate(lines):
            cv2.line(result_frame, line.center, line.centroid, (90, 90, 90), 1)
            cv2.line(original_frame, line.center, line.centroid, (0, 0, 125), 1)
            
            if line.is_tick_mark and line.extended_end:
                ###### CASTRO 25/02/2026 - A LINHA CINZA-CLARO RESSALTA QUAIS REGIÕES SERÃO UTILIZADAS PARA CONTABILIZAR
                cv2.line(result_frame, line.center, line.extended_end, (200, 200, 200), 1)
                
                #if line.circle_intersection:
                    #cv2.circle(result_frame, line.circle_intersection, 3, (255, 255, 0), -1)
                
                ###### CASTRO 25/02/2026 - AGORA VEM O RESULTADO DA RESOLUÇÃO
                cv2.putText(result_frame, str(line.res), line.extended_end_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def find_nearest_region(self, click_x: int, click_y: int, component_centroids: List[Tuple[int, int]], offset: Tuple[int, int], max_distance: int = 50) -> Optional[Tuple[int, int]]:
        ###### CASTRO 25/02/2026 - PARA PODER SELECIONAR O MÍNIMO E MÁXIMO, ADICIONEI UMA REGIÃO CIRCULAR PARA DETECTAR SE ALGUM CENTRÓIDE FOI SELECIONADO. 
        ###### DO CONTRÁRIO, USANDO SOMENTE O PIXEL SELECIONADO, FICAVA DIFÍCIL DE CATAR ALGUMA LINHA; PARA NÃO DAR CONFUSÃO, TALVEZ SEJA UMA BOA DEIXAR UM CÍRCULO EM VOLTA DO CURSOR 
        ###### DO MOUSE NESSA JANELA
        if not component_centroids or offset is None:
            return None
        
        crop_click_x, crop_click_y = click_x - offset[0], click_y - offset[1]
        
        min_dist = float('inf')
        closest_centroid = None
        
        for cx, cy in component_centroids:
            dist = np.sqrt((cx - crop_click_x)**2 + (cy - crop_click_y)**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                closest_centroid = (cx, cy)
        
        return closest_centroid
    
    @staticmethod
    def crop_ring_area(image: np.ndarray, center_x: int, center_y: int, inner_r: int, outer_r: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        
    ###### CASTRO 25/02/2026 - TÁ BOM, AQUI É ONDE A DETECÇÃO DOS PONTOS DE MEDIÇÃO ACONTECEM, NÃO É NADA ESPECIAL, SÓ PRECISA SER PICA PARA PENSAR NISSO
        hh, ww = image.shape[:2]
        white_background = np.ones((hh, ww, 3), dtype=np.uint8) * 255
        
        mask_ring = np.zeros((hh, ww), dtype=np.uint8)
        cv2.circle(mask_ring, (center_x, center_y), outer_r, 255, -1)
        cv2.circle(mask_ring, (center_x, center_y), inner_r, 0, -1)
        
        ring_from_original = cv2.bitwise_and(image, image, mask=mask_ring)
        mask_bg = cv2.bitwise_not(mask_ring)
        white_bg = cv2.bitwise_and(white_background, white_background, mask=mask_bg)
        final_image = cv2.add(ring_from_original, white_bg)

        x1 = max(0, center_x - outer_r)
        y1 = max(0, center_y - outer_r)
        x2 = min(ww, center_x + outer_r)
        y2 = min(hh, center_y + outer_r)
        
        cropped = final_image[y1:y2, x1:x2]
        offset = (x1, y1)
        
        return cropped, offset
    
    @staticmethod
    def enhance_image(frame: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        ###### CASTRO 25/02/2026 - TACA UMA ELEVAÇÃO DE CONTRASTE E BOLA PRA FRENTE
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    @staticmethod
    def convert_to_gray(frame: np.ndarray) -> np.ndarray:
        
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

# ===========
# RING PROCESSOR
# ============================================================================

class RingProcessor:
    def __init__(self, params: ProcessingParams, needle_detector: 'NeedleDetector'):
        self.params = params
        self.needle_detector = needle_detector
        self.app = needle_detector.app
        self.image_processor = ImageProcessor(params, app=self.app)
        
    def process(self, ring_img: np.ndarray, result_frame=None, offset=None) -> List[LineData]:
        
        gray = self.image_processor.convert_to_gray(ring_img)
        _, output_thresh = cv2.threshold(gray, self.params.ring_thresh_low,self.params.ring_thresh_high, cv2.THRESH_BINARY)
        lines = self.image_processor.analyze_ring(ring_img, offset, self.app.gauge_data, result_frame, self.app.frame)
        
        return lines

# ======
# NEEDLE DETECTOR
# ============================================================================

class NeedleDetector:
    def __init__(self, params: ProcessingParams, app: 'GaugeReaderApp' = None):
        self.params = params
        self.app = app
        self.ring_processor = RingProcessor(params, self)
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=params.history,varThreshold=params.var_threshold,detectShadows=True)
    
    def detect_circle(self, gray: np.ndarray, roi_config: Optional[ROIConfig] = None) -> Optional[Tuple[int, int, int]]:
        if roi_config and roi_config.selected:
            return (roi_config.center_x, roi_config.center_y, roi_config.radius)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            return (x, y, int(r * 0.9))
        
        return None
    def extract_needle(self, frame: np.ndarray, circle_data: Tuple[int, int, int], mask: np.ndarray, gauge_data: GaugeData, offset: Optional[Tuple[int, int]] = None) -> Optional[NeedleData]:
        
        x, y, r = circle_data
        
        if offset:
            local_x, local_y, local_r = x - offset[0], y - offset[1], r
        else:
            local_x, local_y, local_r = x, y, r
        
        h, w = frame.shape[:2]
        local_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(local_mask, (local_x, local_y), local_r, 255, -1)
        
        fg_mask_full = self.backSub.apply(frame)
        
        fg_mask_circular =  cv2.bitwise_and(fg_mask_full, fg_mask_full, mask=local_mask)
        
        _, mask_thresh =    cv2.threshold(fg_mask_circular, self.params.thresh_value,self.params.thresh_value2, cv2.THRESH_BINARY)
        
        #cv2.imshow('mask_thresh', mask_thresh)
        
        kernel =            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        mask_eroded =       cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ =       cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.params.min_contour_area]
        
        if not large_contours:
            return None
        
        rectangle_mask = np.zeros_like(mask_thresh)
        all_points = []
        
        for cnt in large_contours:
            x_rect, y_rect, w, h = cv2.boundingRect(cnt)
            contour_center = (x_rect + w//2, y_rect + h//2)
            dist = np.sqrt((contour_center[0] - local_x)**2 + (contour_center[1] - local_y)**2)
            
            if dist <= local_r:
                cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 0, 255), 1)
                cv2.rectangle(rectangle_mask, (x_rect, y_rect), (x_rect + w, y_rect + h), 255, -1)
        
        masked_canny = cv2.bitwise_and(mask_thresh, rectangle_mask)
        indices = np.where(masked_canny > 0)
        
        for px, py in zip(indices[1], indices[0]):
            dist = np.sqrt((px - local_x)**2 + (py - local_y)**2)
            if (py < rectangle_mask.shape[0] and px < rectangle_mask.shape[1] and 
                rectangle_mask[py, px] > 0 and dist <= local_r):
                all_points.append([px, py])
        
        if len(all_points) < 20:
            return None
        
        fitline = np.array(all_points, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(fitline, cv2.DIST_L2, 0, 0.01, 0.01)
        
        if abs(vx[0]) <= 1e-6:
            return None
        
        centroid = (int(x0[0]), int(y0[0]))
        
        dx, dy = centroid[0] - local_x, centroid[1] - local_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance <= 0:
            return None
        
        dx_norm, dy_norm = dx / distance, dy / distance
        
        extension = local_r - distance + 0.1 * local_r
        tip_x, tip_y = int(local_x + dx_norm * (distance + extension)), int(local_y + dy_norm * (distance + extension)) 
        
        intersection_x, intersection_y = int(local_x + dx_norm * local_r), int(local_y + dy_norm * local_r)
        
        # Se tem offset, converte de volta para coordenadas originais para o NeedleData
        if offset:
            abs_centroid = (centroid[0] + offset[0], centroid[1] + offset[1])
            abs_tip = (tip_x + offset[0], tip_y + offset[1])
            abs_intersection = (intersection_x + offset[0], intersection_y + offset[1])
            abs_center = (local_x + offset[0], local_y + offset[1])
        else:
            abs_centroid = centroid
            abs_tip = (tip_x, tip_y)
            abs_intersection = (intersection_x, intersection_y)
            abs_center = (local_x, local_y)
        
        needle = NeedleData( centroid=abs_centroid, tip=abs_tip, circle_intersection=abs_intersection, absolute_angle=self.app.angle_calculator.calculate_angle_from_vector( abs_tip[0] - abs_center[0],  abs_tip[1] - abs_center[1] ) )
        
        if gauge_data.min_line:
            needle_line = (needle.tip, abs_center)
            min_line = (gauge_data.min_line.centroid, gauge_data.min_line.center)
            needle.relative_angle = self.app.angle_calculator.relative_angle_from_min(min_line, needle_line)
        
        return needle
    def process_ring(self, frame, circle_data, result_frame=None) -> List[LineData]:
        
        x, y, r = circle_data
        inner_radius = int(r * self.params.ring_scale_factor)
        
        if inner_radius < r:
            ring_cropped, offset = self.ring_processor.image_processor.crop_ring_area(frame, x, y, inner_radius, r)
            
            if ring_cropped is not None and ring_cropped.size > 0:
                return self.ring_processor.process(ring_cropped, result_frame, offset)
        
        return []

# ===========
# VISUALIZATION
# ============================================================================

class Visualizer:
    
    @staticmethod
    def draw_circle(frame: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int] = (0, 255, 0)):
        cv2.circle(frame, center, radius, color, 1)
        cv2.circle(frame, center, 2, color, 1)
    
    @staticmethod
    def draw_needle(frame: np.ndarray, needle: NeedleData, color: Tuple[int, int, int] = (0, 255, 255)):
        if needle.tip and needle.centroid:
            cv2.line(frame, needle.centroid, needle.tip, color, 2)
            cv2.circle(frame, needle.centroid, 5, (0, 255, 0), 1)
    
    @staticmethod
    def draw_intersection_marker(frame: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 255)):
        if point:
            cv2.drawMarker(frame, point, color, cv2.MARKER_SQUARE, 20, 1, 1)
    
    @staticmethod
    def draw_calibration_lines(frame: np.ndarray, gauge_data: GaugeData):
        ###### CASTRO 25/02/2026 - TÁ AQUI ELE CONSULTA O A LISTA DE MIN_LNE E MAX_LINE PARA DESENHAR NO FRAME A CADA ITERAÇÃO
        if gauge_data.min_line:
            cv2.line(frame, gauge_data.min_line.center, gauge_data.min_line.centroid, (0, 255, 0), 1)
            cv2.circle(frame, gauge_data.min_line.centroid, 5, (0, 255, 0), -1)
            cv2.putText(frame,str(gauge_data.min_value), (gauge_data.min_line.centroid[0]-35,gauge_data.min_line.centroid[1]+25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0), 1)
        
        if gauge_data.max_line:
            cv2.line(frame, gauge_data.max_line.center, gauge_data.max_line.centroid, (0, 0, 255), 1)
            cv2.circle(frame, gauge_data.max_line.centroid, 5, (0, 0, 255), -1)
            ###### CASTRO 25/02/2026 - TIREI PORQUE TEM UMA FUNÇÃO JÁ ITERANDO PARA ISSO
            #cv2.putText(frame,str(gauge_data.max_value), (gauge_data.max_line.centroid[0] + 35,gauge_data.max_line.centroid[1] + 25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255), 1)
    
    @staticmethod
    def draw_permanent_lines(frame: np.ndarray, gauge_data: GaugeData):
        ###### CASTRO 25/02/2026 - DESENHA AS LINHAS PERMANENTES CAPTURADAS APÓS CALIBRAÇÃO
        if gauge_data.permanent_lines:
            for line in gauge_data.permanent_lines:
                cv2.line(frame, line.center, line.centroid, (90, 90, 90), 1)
                
                if line.is_tick_mark and line.extended_end:
                    cv2.line(frame, line.center, line.extended_end, (200, 200, 200), 1)
                    
                    #if line.circle_intersection:
                        #cv2.circle(frame, line.circle_intersection, 3, (255, 255, 0), -1)
                    
                    cv2.putText(frame, str(line.res), line.extended_end_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    @staticmethod
    def draw_display_info(frame: np.ndarray, gauge_data: GaugeData, state: ProcessingState):
        
        if gauge_data.current_read > 0:
            cv2.putText(frame, str(f'{gauge_data.current_read} [{gauge_data.unit}]'), (35, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 2)
            
        else:
            cv2.putText(frame, str(f''), (35, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), 2)

        if gauge_data.warning_max_instant == True:
            cv2.putText(frame, str(f'Warning'), (35, 225), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 2)

    @staticmethod
    def draw_info_overlay(frame: np.ndarray, gauge_data: GaugeData, state: ProcessingState):
        
        y_pos = 30
        info_items = [("frame", f"{state.frame_counter}", "", y_pos),("abs(needle_angle)", f"{gauge_data.needle.absolute_angle:.1f}", "deg", y_pos + 30),]
        
        if gauge_data.min_line:
            info_items.append(("relative_needle_angle", f"{gauge_data.needle.relative_angle:.1f}", "deg", y_pos + 60))
        
        info_items.append(("current_read",                  f"{int(gauge_data.current_read)}",          "unit",     y_pos + 230))
        info_items.append(("min_value",                     f"{int(gauge_data.min_value)}",             "unit",     y_pos + 260))
        info_items.append(("max_value",                     f"{int(gauge_data.max_value)}",             "unit",     y_pos + 290))
        info_items.append(("resolution",                    f"{int(gauge_data.res)}",                   "unit",     y_pos + 320))
        info_items.append(("bool(warning_max_instant)",     f"{str(gauge_data.warning_max_instant)}",   "",         y_pos + 350))
        info_items.append(("bool(warning_max_flag)",        f"{str(gauge_data.warning_max_flag)}",      "",         y_pos + 380))
        
        for label, value, unit, y in info_items:
            text = f"{label}: {value}"
            if unit:
                text += f" [{unit}]"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), )
        cv2.rectangle(frame, (5,240),(200,270),(0,0,255),2)
    
    @staticmethod
    def draw_calibration_mode(frame: np.ndarray, stage: CalibrationStage):
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        if stage == CalibrationStage.SELECTING_MIN:
            cv2.putText(frame, "min_flag", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        elif stage == CalibrationStage.SELECTING_MAX:
            cv2.putText(frame, "max_flag", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# ==============
# MAIN APPLICATION
# ============================================================================

class GaugeReaderApp:

    def __init__(self, video_path: str):
        self.video_path =           video_path
        self.params =               ProcessingParams()
        self.roi_config =           ROIConfig()
        self.state =                ProcessingState()

        self.calibrator =           CameraCalibrator(CameraParams())
        self.angle_calculator =     AngleCalculator(self.params.value_range, self.params.resolution)
        self.image_processor =      ImageProcessor(self.params, app=self)
        self.needle_detector =      NeedleDetector(self.params, app=self)
        self.visualizer =           Visualizer()

        self.gauge_data =           GaugeData()

        self.cap =                  None
        self.total_frames =         0
        self.fps =                  0
        self.paused =               False
        
        self.current_result_frame = None
        self.frame =                None
        self.circle_data =          None

        self.last_needle_angle =    0

    def process_frozen_frame_for_calibration(self):
        
        if self.state.frozen_frame is None:
            print(">>> NO FRAME AVAILABLE")
            return False
        
        gray = self.image_processor.convert_to_gray(self.state.frozen_frame)
        circle_data = self.needle_detector.detect_circle(gray, self.roi_config)
        
        if not circle_data:
            print(">>> COULD NOT DETECT CIRCLE IN FROZEN FRAME")
            return False
        
        x, y, r = circle_data
        self.gauge_data.center =            (x, y)
        self.gauge_data.radius =            r
        self.gauge_data.circle_detected =   True
        
        temp_result = np.zeros_like(self.state.frozen_frame)
        
        inner_radius = int(r * self.params.ring_scale_factor)
        if inner_radius < r:
            ring_cropped, offset = self.image_processor.crop_ring_area(self.state.frozen_frame, x, y, inner_radius, r)
            
            if ring_cropped is not None and ring_cropped.size > 0:
                lines = self.image_processor.analyze_ring(ring_cropped, offset, self.gauge_data, temp_result, self.state.frozen_frame)
                self.gauge_data.detected_lines = lines
                
                print(f">>> len_lines: {len(lines)}")
                return len(lines) > 0
        
        return False

    def enter_calibration_mode(self):
        if self.frame is not None:
            ###### CASTRO 25/02/2026 - LIMPA TUDO PARA NOVA CALIBRAÇÃO
            self.gauge_data.permanent_lines.clear()
            self.gauge_data.lines_captured =        False
            
            self.gauge_data.min_line =              None
            self.gauge_data.max_line =              None
            self.gauge_data.is_calibrated =         False
            
            self.gauge_data.tick_mark_angles.clear()
            self.gauge_data.scale_values.clear()
            self.gauge_data.crossed_angles.clear()
            self.gauge_data.max_crossed =           False
            self.gauge_data.warning_max_flag =      False
            self.gauge_data.warning_max_instant =   False
            self.gauge_data.current_read =          0.0
            self.gauge_data.total_angle_span =      0.0
            
            self.last_needle_angle =                0

            self.state.frozen_frame =               self.frame.copy()
            self.state.calibration_mode =           True
            self.state.calibration_stage =          CalibrationStage.SELECTING_MIN
            self.paused =                           True
            
            success = self.process_frozen_frame_for_calibration()

    def exit_calibration_mode(self):
        self.state.calibration_mode = False
        self.state.calibration_stage = CalibrationStage.NOT_CALIBRATING
        self.paused = False
        print(">>> CALIBRATION CANCELLED")

    def capture_permanent_lines(self):
        ###### CASTRO 25/02/2026 - CAPTURA AS LINHAS UMA ÚNICA VEZ APÓS CALIBRAÇÃO
        if self.frame is not None and self.gauge_data.circle_detected:
            enhanced_frame =    self.image_processor.enhance_image(self.frame, self.params.alpha, self.params.beta)
            circle_data =       (self.gauge_data.center[0], self.gauge_data.center[1], self.gauge_data.radius)
            
            lines = self.needle_detector.process_ring(enhanced_frame, circle_data, None)
            
            if lines:
                self.gauge_data.permanent_lines = lines
                self._update_permanent_lines_angles()
                self.gauge_data.lines_captured = True
                #print(f"Linhas permanentes capturadas: {len(lines)}")
                return True
        return False

    def _update_permanent_lines_angles(self):
        ###### CASTRO 25/02/2026 - CALCULA ÂNGULOS E RES PARA AS LINHAS PERMANENTES
        if not self.gauge_data.min_line or not self.gauge_data.permanent_lines:
            return
        
        min_line_tuple =                (self.gauge_data.min_line.centroid, self.gauge_data.min_line.center)
        
        for line in self.gauge_data.permanent_lines:
            line_tuple =                (line.centroid, line.center)
            line.angle_from_min =       self.angle_calculator.relative_angle_from_min(min_line_tuple, line_tuple)
            
            if self.gauge_data.max_line:
                max_angle =             self.gauge_data.max_line.angle_from_min
                max_range =             max_angle
                ##### 26/02/2026 CASTRO - AQUI FIXEI UMA CONDIÇÃO QUE ACHO MEEIO SUSPEITA
                ##### EM RELAÇÃO AO ÂNGULO, A LINHA VAI SER CONSIDERADA OU NÃO
                ##### TALVEZ NO FUTURO TENHA DE REVER ESTE CRITÉRIO PARA OUTROS MANÔMETROS
                angle_tolerance = (max_range - line.angle_from_min) * 0.01
                line.is_tick_mark = (angle_tolerance <= line.angle_from_min <= max_range + angle_tolerance)
                ##### ANTES ESTAVA ASSIM, NÃO SEI SE VAI ROLAR
                #line.is_tick_mark =     (3 <= line.angle_from_min <= max_range + 3)
        
        tick_lines =            [line for line in self.gauge_data.permanent_lines if line.is_tick_mark]
        tick_lines.sort(key=lambda l: l.angle_from_min)
        
        for idx, line in enumerate(tick_lines):
            line.res =          self.gauge_data.min_value + (idx + 1) * self.gauge_data.res
            
        self.gauge_data.update_tick_marks()

    def handle_calibration_click(self, x: int, y: int):
        if not self.state.calibration_mode:
            return
        
        if not hasattr(self.image_processor, 'last_component_centroids'):
            print(">>> NO REGION DATA AVAILABLE. PROCESSING FRAME AGAIN")
            self.process_frozen_frame_for_calibration()
            return
        
        component_centroids =   self.image_processor.last_component_centroids
        offset =                self.image_processor.last_offset
        
        if not component_centroids or offset is None:
            print(">>> NO REGIONS DETECTED IN THIS FRAME")
            return
        
        nearest =               self.image_processor.find_nearest_region(x, y, component_centroids, offset)
        
        if nearest is None:
            print(">>> NO REGION FOUND NEAR CLICK.")
            return
        
        original_x, original_y = offset[0] + nearest[0], offset[1] + nearest[1]
        
        target_line = None
        for line in self.gauge_data.detected_lines:
            if (abs(line.centroid[0] - original_x) < 5 and abs(line.centroid[1] - original_y) < 5):
                target_line = line
                break
        
        if not target_line:
            print(">>> COULD NOT FIND MATCHING LINE DATA")
            return
        
        if self.state.calibration_stage ==          CalibrationStage.SELECTING_MIN:
            self.gauge_data.min_line =              target_line
            target_line.is_tick_mark =              True
            target_line.angle_from_min =            0 
            self.state.calibration_stage =          CalibrationStage.SELECTING_MAX
            print(f">>> min_line: ({original_x}, {original_y})")
            
        elif self.state.calibration_stage ==        CalibrationStage.SELECTING_MAX:
            self.gauge_data.max_line =              target_line
            target_line.is_tick_mark =              True
            
            if self.gauge_data.min_line:
                min_line_tuple =                    (self.gauge_data.min_line.centroid, self.gauge_data.min_line.center)
                max_line_tuple =                    (target_line.centroid, target_line.center)
                max_angle =                         self.angle_calculator.relative_angle_from_min(min_line_tuple, max_line_tuple)
                target_line.angle_from_min =        max_angle
                self.gauge_data.total_angle_span =  max_angle
            
            self.gauge_data.is_calibrated =         True
            self.state.calibration_mode =           False
            self.state.calibration_stage =          CalibrationStage.NOT_CALIBRATING
            self.paused =                           False
            
            ###### CASTRO 25/02/2026 - APÓS CALIBRAÇÃO, CAPTURA AS LINHAS PERMANENTES
            self.capture_permanent_lines()
            
            print(f">>> max_line ({original_x}, {original_y})")
            print(f">>> angle_span: {self.gauge_data.total_angle_span:.1f}°")
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        
        enhanced_frame =            self.image_processor.enhance_image(frame, self.params.alpha, self.params.beta)
        gray =                      self.image_processor.convert_to_gray(enhanced_frame)
        
        black_foreground =          np.zeros_like(frame)
        display =                   np.zeros((400, 850, 3),dtype=np.uint8)
        self.current_result_frame = black_foreground
        self.frame =                frame
        
        mask =                      np.zeros(frame.shape[:2], dtype=np.uint8)

        circle_data =               self.needle_detector.detect_circle(gray, self.roi_config)
        
        if not circle_data:
            return None
        
        x, y, r = circle_data
        self.gauge_data.center = (x, y)
        self.gauge_data.radius = r
        self.gauge_data.circle_detected = True
        
        self.visualizer.draw_circle(frame, (x, y), r)
        self.visualizer.draw_circle(black_foreground, (x, y), r)
        
        ###### CASTRO 25/02/2026 - AGORA USA AS LINHAS PERMANENTES SE EXISTIREM
        if self.gauge_data.lines_captured and self.gauge_data.permanent_lines:
            self.visualizer.draw_permanent_lines(black_foreground, self.gauge_data)
            self.visualizer.draw_permanent_lines(frame, self.gauge_data)

        if not self.state.calibration_mode:
            ###### CASTRO 26/02/2026 - DALE AGORA ISOLEI O VERSO DO PONTEIRO PARA NÃO DAR FALSO-POSITIVO NA MEDIÇÃO
            ###### CASTRO 26/02/2026 - EM ALGUMA HORA ESSAS CONSTANTES TEM DE VIRAR UM PARÂMETRO
            inner_radius = int(r * 0.5)
            outer_radius = int(r + r * 0.2)
            
            ring_cropped, offset = self.image_processor.crop_ring_area(enhanced_frame, x, y, inner_radius, outer_radius)
            #cv2.imshow('ring_cropped', ring_cropped)
            
            needle = self.needle_detector.extract_needle(ring_cropped, circle_data, mask, self.gauge_data, offset)
                #cv2.imshow('enhanced_frame',enhanced_frame)
            if needle:
                self.gauge_data.needle = needle
                
                self.visualizer.draw_needle(black_foreground, needle)
                self.visualizer.draw_intersection_marker(black_foreground, needle.circle_intersection)
                cv2.putText(black_foreground, f"current needle: {needle.relative_angle:.1f} [deg]", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
                if self.gauge_data.is_calibrated and self.gauge_data.tick_mark_angles:
                    
                    ###### CASTRO 25/02/2026 - ENCONTRA O VALOR CORRETO BASEADO NA POSIÇÃO DO PONTEIRO
                    tick_lines = [line for line in self.gauge_data.permanent_lines if line.is_tick_mark]
                    tick_lines.sort(key=lambda l: l.angle_from_min)
                    
                    current_value = self.gauge_data.min_value  
                    
                    for i, line in enumerate(tick_lines):
                        if needle.relative_angle >= line.angle_from_min:
                            current_value = line.res
                        else:
                            break
                    
                    self.gauge_data.current_read = current_value  
                    
                    for tick_angle in self.gauge_data.tick_mark_angles:
                        if needle.relative_angle > tick_angle:
                            if not hasattr(self, 'last_needle_angle'):
                                self.last_needle_angle = 0
                            
                            if needle.relative_angle < max(self.gauge_data.tick_mark_angles) - 15:
                                self.gauge_data.warning_max_instant = False 
                            
                            if self.last_needle_angle <= tick_angle:
                                if tick_angle not in self.gauge_data.crossed_angles:
                                    self.gauge_data.crossed_angles[tick_angle] = 1
                                else:
                                    self.gauge_data.crossed_angles[tick_angle] += 1
    
                                if (tick_angle + 10) > max(self.gauge_data.tick_mark_angles):
                                    self.gauge_data.warning_max_flag = True
                                    self.gauge_data.warning_max_instant = True
                    
                    self.last_needle_angle = needle.relative_angle
        
        self.visualizer.draw_calibration_lines(black_foreground, self.gauge_data)
        self.visualizer.draw_calibration_lines(frame, self.gauge_data)
        
        if self.state.calibration_mode:
            self.visualizer.draw_calibration_mode(black_foreground, self.state.calibration_stage)
        
        self.visualizer.draw_info_overlay(black_foreground, self.gauge_data, self.state)
        self.visualizer.draw_display_info(display, self.gauge_data, self.state)
        
        return black_foreground, display
    
    def initialize_video(self):
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f">>> COULD NOT OPEN VIDEO FILE {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def setup_windows(self):
        
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self._mouse_callback)
        
        #cv2.namedWindow('CONTROLS')
        #cv2.createTrackbar('thresh_value',      'CONTROLS',         self.params.thresh_value,           255,    lambda v: setattr(self.params,         'thresh_value',     v))
        #cv2.createTrackbar('thresh_value2',     'CONTROLS',         self.params.thresh_value2,          255,    lambda v: setattr(self.params,         'thresh_value2',    v))
        #cv2.createTrackbar('H_varThreshold',    'CONTROLS',         self.params.var_threshold,          255,    lambda v: setattr(self.params,         'var_threshold',    v))
        #cv2.createTrackbar('ROI Radius',        'CONTROLS',         self.roi_config.radius,             300,    lambda v: setattr(self.roi_config,     'radius',           v))

        #cv2.namedWindow('output_thresh')    
        #cv2.createTrackbar('ring_thresh_low',   'output_thresh',    self.params.ring_thresh_low,        255,    lambda v: setattr(self.params,         'ring_thresh_low',  v))
        #cv2.createTrackbar('ring_thresh_high',  'output_thresh',    self.params.ring_thresh_high,       255,    lambda v: setattr(self.params,         'ring_thresh_high', v))
        #cv2.createTrackbar('N regions',         'output_thresh',    self.params.n_regions,              100,    lambda v: setattr(self.params,         'n_regions',        max(1, v)))
        
        #cv2.namedWindow('detected_regions')

    def _mouse_callback(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state.calibration_mode:
                self.handle_calibration_click(x, y)
            else:
                self.roi_config.center_x, self.roi_config.center_y = x, y
                print(f">>> CIRCLE CENTER SET TO ({x}, {y})")

    ##### CASTRO 26/02/2026 - CENSUREI MAS É MELHOR DEIXAR POR AÍ
    def register_position(self):
        pass

    ##### CASTRO 26/02/2026 - CENSUREI MAS É MELHOR DEIXAR POR AÍ
    def calculate_all_angles(self):
        pass

    def run(self):
        
        self.initialize_video()
        self.setup_windows()

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LANCZOS4)
                frame = self.calibrator.undistort(frame)
                original_frame = frame.copy()
            else:
                original_frame = self.frame if self.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            result_frame, display = self.process_frame(original_frame)
            
            if result_frame is not None:
                cv2.imshow("app", result_frame)
                cv2.imshow("DISPLAY", display)
            
            cv2.imshow('frame', original_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n>>> EXITING APPLICATION")
                break
            elif key == 32:  
                self.register_position()
            elif key == ord('c'):
                self.calculate_all_angles()
            elif key == ord('p'):
                self.paused = not self.paused
                print(f">>> VIDEO {'PAUSEE' if self.paused else 'RESUME'}")
            elif key == ord('r'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.state.frame_counter = 0
                self.gauge_data.reset_tracking()
                print(">>> VIDEO AND TRACKING RESET")
            elif key == ord('m'): 
                self.enter_calibration_mode()
            elif key == ord('x'):  
                self.gauge_data.reset_tracking()
                print(">>> CROSSING TARGETS RESET")
            elif key == 27:  
                if self.state.calibration_mode:
                    self.exit_calibration_mode()
                else:
                    print("\n>>> EXITING APPLICATION")
                    break
            
            self.state.frame_counter += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close('all')
        print("\n>>> APPLICATION CLOSED")

# ===========
# É AQUI QUE COMEÇA
# ============================================================================

if __name__ == "__main__":
    video_path = r'record 0 new.mp4'
    app = GaugeReaderApp(video_path)
    app.run()

import os
import sys
import yaml
import time
import torch
import json
import signal
import random
import shutil
import atexit
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from generate_training_report import generate_training_report

# 設置隨機種子以確保結果可重現
random.seed(42)

# 定義路徑
BASE_DIR = Path(__file__).parent  # 基礎目錄
DATA_DIR = BASE_DIR / 'aortic_valve_data'  # 數據目錄

# 創建必要的目錄
DATA_DIR.mkdir(exist_ok=True, parents=True)

# 獲取病人列表
patients = sorted([d for d in os.listdir('training_image') if os.path.isdir(os.path.join('training_image', d))])

# 將病人分為訓練集和驗證集（80% 訓練，20% 驗證）
train_patients, val_patients = train_test_split(patients, test_size=0.2, random_state=42)

def prepare_data(patients, split_name):
    """
    準備 YOLO 格式的圖像和標籤
    
    參數:
        patients: 病人ID列表
        split_name: 數據集分割名稱（'train' 或 'val'）
    """
    split_dir = DATA_DIR / split_name  # 分割目錄
    img_dir = split_dir / 'images'    # 圖像目錄
    label_dir = split_dir / 'labels'  # 標籤目錄
    
    # 創建圖像目錄（如果不存在）
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"準備 {split_name} 數據中...")
    
    for patient in tqdm(patients, desc=f"處理 {split_name}"):
        # 處理圖像
        img_src = Path('training_image') / patient
        for img_file in img_src.glob('*.png'):
            # 複製圖像
            img_dest = img_dir / f"{patient}_{img_file.name}"
            if not img_dest.exists():
                shutil.copy(img_file, img_dest)
            
            # 處理對應的標籤
            label_name = f"{patient}_{img_file.stem.split('_')[-1]}.txt"
            label_src = Path('training_label') / patient / label_name
            
            if label_src.exists():
                # 複製標籤
                label_dest = label_dir / f"{patient}_{img_file.stem}.txt"
                shutil.copy(label_src, label_dest)
    
    print(f"{split_name} 數據準備完成。")
    return len(list(img_dir.glob('*.png')))

# 準備訓練和驗證數據
train_count = prepare_data(train_patients, 'train')
val_count = prepare_data(val_patients, 'val')

print(f"\n數據集準備完成！")
print(f"訓練樣本數: {train_count}")
print(f"驗證樣本數: {val_count}")

# 創建 YAML 配置文件
data_yaml = {
    'path': str(DATA_DIR.absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'names': {
        0: 'aortic_valve',
    },
    'nc': 1,  # 類別數量
    'download': None
}

# 保存 YAML 文件
yaml_path = DATA_DIR / 'aortic_valve.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=None, sort_keys=False)

print(f"\n數據集配置已保存至: {yaml_path}")

LOCK_PATH = BASE_DIR / 'train.lock'

def acquire_lock():
    if LOCK_PATH.exists():
        try:
            pid = int(LOCK_PATH.read_text().strip())
            if pid and psutil.pid_exists(pid):
                print(f"另一個訓練程序正在運行 (PID {pid})。正在退出。")
                raise SystemExit(0)
        except Exception:
            pass
    LOCK_PATH.write_text(str(os.getpid()))
    atexit.register(release_lock)

def release_lock():
    try:
        if LOCK_PATH.exists():
            LOCK_PATH.unlink()
    except Exception:
        pass

class TrainingMonitor:
    def __init__(self, log_file='training_progress.json'):
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self.log_file = Path(log_file)
        self.status = {
            'status': 'starting',
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_epoch': 0,
            'total_epochs': 150,
            'elapsed_time': '00:00:00',
            'eta': '--:--:--',
            'metrics': {},
            'last_update': ''
        }
        self._save_status()
    
    def update(self, epoch, metrics=None):
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.status['elapsed_time'] = str(timedelta(seconds=int(elapsed)))
        
        if metrics:
            self.status['metrics'] = metrics
        
        self.status['current_epoch'] = epoch
        
        # 計算 ETA
        if epoch > 1:
            avg_time_per_epoch = elapsed / epoch
            remaining_epochs = self.status['total_epochs'] - epoch
            eta_seconds = int(avg_time_per_epoch * remaining_epochs)
            self.status['eta'] = str(timedelta(seconds=eta_seconds))
        
        # 每小時或每個 epoch 都保存狀態
        if current_time - self.last_report_time >= 3600 or epoch == self.status['total_epochs']:
            self.last_report_time = current_time
            self.status['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self._save_status()
            self._print_status()
    
    def _save_status(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.status, f, ensure_ascii=False, indent=2)
    
    def _print_status(self):
        print("\n" + "="*50)
        print(f"訓練狀態更新 ({self.status['last_update']})")
        print("-"*50)
        print(f"當前狀態: {self.status['status']}")
        print(f"進度: {self.status['current_epoch']}/{self.status['total_epochs']} epochs")
        print(f"已用時間: {self.status['elapsed_time']}")
        print(f"預計剩餘時間: {self.status['eta']}")
        
        if self.status['metrics']:
            print("\n當前指標:")
            for k, v in self.status['metrics'].items():
                print(f"  {k}: {v:.4f}")
        print("="*50 + "\n")
    
    def complete(self, success=True, error_msg=None):
        self.status['status'] = 'completed' if success else 'failed'
        if error_msg:
            self.status['error'] = error_msg
        self.status['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_status()
        self._print_status()

def signal_handler(signum, frame):
    print("\n收到中斷信號，正在保存訓練狀態...")
    if 'monitor' in globals():
        monitor.complete(success=False, error_msg="訓練被用戶中斷")
    sys.exit(1)

def main():
    global monitor
    
    # 註冊信號處理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        acquire_lock()
        start_time = time.time()
        
        # 初始化監控器
        monitor = TrainingMonitor('training_status.json')
        
        # 訓練配置
        training_config = {
            # 基本設置
            'data': str(yaml_path),      # 數據配置文件路徑
            'epochs': 100,               # 訓練總輪數（增加訓練輪數）
            'imgsz': 832,                # 輸入圖像大小（像素）
            'batch': 8,                  # 每個批次的圖像數量（YOLOv10m較大，需要減少批次大小）
            'device': 0,                 # 使用第一個GPU（0表示GPU:0）
            'workers': 4,                # 數據加載的工作進程數
            'project': 'aortic_valve_detection',  # 項目名稱
            'name': 'yolov10m_aortic_valve_v1',  # 模型版本名稱
            'exist_ok': True,            # 允許覆蓋現有項目
            
            # 優化器配置
            'optimizer': 'AdamW',         # 優化器類型：AdamW
            'lr0': 0.008,                # 初始學習率（比YOLOv10n略小）
            'lrf': 0.01,                 # 最終學習率倍數（最終學習率 = lr0 * lrf）
            'momentum': 0.9,             # 優化器動量參數（AdamW通常設為0.9）
            'weight_decay': 0.0003,      # 權重衰減係數（L2正則化，比YOLOv10n略小）
            'warmup_epochs': 3.0,        # 學習率熱身階段的輪數
            'warmup_momentum': 0.8,      # 熱身階段的動量值
            'warmup_bias_lr': 0.1,       # 熱身階段的偏置項學習率
            
            # 數據增強配置
            # 顏色增強
            'hsv_h': 0.015,              # 色調增強幅度（0-0.5）
            'hsv_s': 0.7,                # 飽和度增強幅度（0-1）
            'hsv_v': 0.4,                # 明度增強幅度（0-1）
            
            # 幾何變換
            'degrees': 15.0,             # 隨機旋轉角度範圍（-degrees, +degrees）
            'translate': 0.1,            # 圖像平移比例（相對於圖像尺寸）
            'scale': 0.2,                # 縮放比例範圍（1-scale, 1+scale）
            'shear': 0.0,                # 剪切強度（度數）
            'perspective': 0.0001,       # 透視變換參數（0-0.001）
            
            # 翻轉增強
            'flipud': 0.5,               # 上下翻轉概率（0-1）
            'fliplr': 0.5,               # 左右翻轉概率（0-1）
            
            # 高級增強
            'mosaic': 1.0,               # Mosaic數據增強概率（0-1）
            'mixup': 0.1,                # MixUp數據增強概率（0-1）
            'copy_paste': 0.1,           # 複製粘貼增強概率（0-1）
            'erasing': 0.4,              # 隨機擦除概率（0-1）
            
            # 損失函數配置
            'fl_gamma': 0.0,             # Focal Loss的gamma參數（0.0表示禁用Focal Loss）
            'label_smoothing': 0.1,      # 標籤平滑係數（0.0-1.0）
            'box': 0.05,                 # 邊界框回歸損失權重
            'cls': 0.5,                  # 分類損失權重
            'dfl': 1.5,                  # Distribution Focal Loss權重
            'fl_alpha': 0.5,             # Focal Loss的alpha參數（0.0表示禁用Focal Loss）
            
            # 訓練控制
            'close_mosaic': 10,          # 在最後N個epoch關閉Mosaic增強
            'nbs': 64,                   # 名義批次大小（用於梯度累積）
            'single_cls': False,         # 是否使用單類別模式（False表示多類別）
            'cos_lr': True,              # 是否使用余弦退火學習率調度
            'amp': True,                 # 是否啟用自動混合精度訓練（減少顯存使用）
            
            # 遮罩相關（實例分割）
            'overlap_mask': True,        # 是否允許遮罩重疊
            'mask_ratio': 4,             # 遮罩降採樣比例（相對於圖像）
            
            # 其他設置
            'fraction': 1.0,             # 訓練數據使用比例（0.0-1.0）
            'profile': False,            # 是否分析訓練速度（會降低訓練速度）
            'deterministic': True,       # 是否啟用確定性模式（確保可重現性）
            'save_period': -1,           # 保存檢查點的間隔（-1表示不保存中間檢查點）
            'seed': 42,                  # 隨機種子（確保實驗可重現）
        }

        print("\n開始訓練，配置如下：")
        for k, v in training_config.items():
            print(f"  {k}: {v}")

        # 初始化 YOLOv10 模型並移動到 GPU
        print("\n開始訓練 YOLOv10 模型...")
        model = YOLO('yolov10m.pt').to('cuda:0')
        
        # 自定義回調類來監控訓練進度
        class TrainingCallback:
            def __init__(self, monitor):
                self.monitor = monitor
            
            def on_train_epoch_end(self, trainer):
                try:
                    metrics = {
                        'train/loss': trainer.toss if hasattr(trainer, 'toss') else 0,  # 修正這裡使用正確的屬性
                        'val/mAP50': getattr(trainer.metrics, 'map50', 0),
                        'val/precision': getattr(trainer.metrics, 'precision', 0),
                        'val/recall': getattr(trainer.metrics, 'recall', 0)
                    }
                    self.monitor.update(trainer.epoch, metrics)
                except Exception as e:
                    print(f"警告: 更新訓練監控時出錯: {e}")
        
        # 創建回調實例
        training_callback = TrainingCallback(monitor)
        
        # 開始訓練
        monitor.status['status'] = 'training'
        
        # 添加回調到模型
        model.add_callback('on_train_epoch_end', training_callback.on_train_epoch_end)
        
        # 開始訓練
        results = model.train(**training_config)
        
        # 訓練完成
        end_time = time.time()
        training_time = str(timedelta(seconds=int(end_time - start_time)))
        
        print("\n" + "="*50)
        print(f"訓練完成！總用時: {training_time}")
        print("="*50)
        
        # 生成訓練報告
        print("\n生成訓練報告...")
        info = generate_training_report()
        
        if info:
            print("\n報告已生成：")
            print(f"  報告: {info['report_path']}")
            print(f"  最佳權重: {info['best_weights']}")
            print(f"  最終權重: {info['last_weights']}")
        
        # 更新監控器狀態
        monitor.complete(success=True)
        
        # 模型評估
        print("\n開始模型評估...")
        metrics = model.val()
        print("\n評估結果:")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  Precision: {metrics.box.precision.mean():.4f}")
        print(f"  Recall: {metrics.box.recall.mean():.4f}")
        
        # 保存評估結果
        eval_results = {
            'mAP50-95': float(metrics.box.map),
            'mAP50': float(metrics.box.map50),
            'precision': float(metrics.box.precision.mean()),
            'recall': float(metrics.box.recall.mean()),
            'training_time': training_time,
            'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n評估結果已保存至: evaluation_results.json")
        
    except Exception as e:
        error_msg = f"訓練過程中發生錯誤: {str(e)}"
        print(f"\n錯誤: {error_msg}")
        if 'monitor' in globals():
            monitor.complete(success=False, error_msg=error_msg)
        raise
    finally:
        # 釋放鎖定
        release_lock()
        # 釋放 GPU 內存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

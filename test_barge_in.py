# test_barge_in.py
import sys
import time
from config import Config
from services.io_service import AudioHandler, WakeWord

def test_interrupt():
    print("=== 打断功能专项测试 ===")
    Config.setup_env_proxy()
    
    # 1. 初始化
    print("1. 初始化组件...")
    ear_mouth = AudioHandler()
    wake_word = WakeWord()
    
    # 2. 测试文本
    long_text = "这是一段非常非常长的测试文本，目的是为了让你有足够的时间喊出Jarvis来打断我。如果你不打断，我会一直念下去，念到地老天荒，念到海枯石烂。请现在大声喊出唤醒词！"
    
    print(f"2. 开始播放长文本: '{long_text[:20]}...'")
    print("   >>> 请在播放期间大声喊 'Jarvis' <<<")

    # 3. 开启麦克风流
    wake_word.start()
    
    # 4. 定义回调
    def check():
        # 打印一个点，证明在检测
        print(".", end="", flush=True)
        return wake_word.process_frame()

    # 5. 开始播放并检测
    start_time = time.time()
    interrupted = ear_mouth.speak(long_text, interrupt_check_callback=check)
    end_time = time.time()
    
    # 6. 结果
    wake_word.stop()
    print("\n")
    if interrupted:
        print(f"✅ 测试成功！在 {end_time - start_time:.2f} 秒时检测到打断。")
    else:
        print("❌ 测试失败：播放完了全句，未检测到打断。")
        print("   可能原因：")
        print("   1. 说话声音太小，被音箱声音盖住了（尝试戴耳机测试）。")
        print("   2. 麦克风被占用。")

    # 清理
    ear_mouth.close()
    wake_word.close()

if __name__ == "__main__":
    test_interrupt()
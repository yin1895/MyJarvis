# services/io_service.py
import os
import asyncio
import tempfile
import struct
import winsound
import wave
from collections import deque
from typing import Any, cast
import numpy as np
import torch
import pvporcupine
import pyaudio
import pygame
import edge_tts
from faster_whisper import WhisperModel
from config import Config

class AudioHandler:
    def __init__(self):
        self.voice = Config.TTS_VOICE
        self.output_file = "reply.mp3"
        
        # 【修复点 1】必须调用 pygame.init() 而不是只 init mixer
        # 否则 pygame.event.pump() 会报错
        try:
            pygame.init() 
            pygame.mixer.init()
        except Exception as e:
            print(f"[Audio Error] Pygame init failed: {e}")

        # 初始化 Faster-Whisper (本地 GPU)
        try:
            # 使用 large-v3 或 medium，根据显存决定
            # compute_type="float16" 利用 GPU 加速
            self.stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
            print("[Init] Faster-Whisper (GPU) Loaded.")
        except Exception as e:
            print(f"[Init Error] Faster-Whisper Load Failed: {e}")
            self.stt_model = None
        
        # 加载 Silero VAD 模型
        try:
            # 使用 torch.hub 加载，trust_repo=True 允许执行 hubconf.py
            # 修复编译器报错：显式类型转换
            loaded_obj = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
            
            if isinstance(loaded_obj, tuple):
                self.vad_model = cast(Any, loaded_obj[0])
            else:
                self.vad_model = cast(Any, loaded_obj)
                
            cast(Any, self.vad_model).eval() # 切换到评估模式
            print("[Init] Silero VAD Loaded.")
        except Exception as e:
            print(f"[Init Error] VAD Load Failed: {e}")
            self.vad_model = None

    def play_ding(self):
        try: winsound.Beep(1000, 200)
        except: pass

    async def _generate_audio(self, text):
        communicate = edge_tts.Communicate(text, self.voice, rate="+20%")
        await communicate.save(self.output_file)

    def speak(self, text, interrupt_check_callback=None):
        # 这里的 print 可以保留用于调试，或者注释掉
        # print(f"[Alice]: {text}", flush=True) 
        
        try:
            if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except: pass

        if os.path.exists(self.output_file):
            try: os.remove(self.output_file)
            except: pass

        try:
            asyncio.run(self._generate_audio(text))
            if not os.path.exists(self.output_file): return

            pygame.mixer.music.load(self.output_file)
            pygame.mixer.music.play()
            
            # 【修复点 2】移除 Clock，完全依赖 mic 读取的阻塞特性来控制循环速度
            # 这样可以最大化利用 CPU 时间进行检测
            while pygame.mixer.music.get_busy():
                # 处理 Windows 事件防止卡死 (现在 init 修复了，这行不会报错了)
                pygame.event.pump()

                if interrupt_check_callback:
                    # 检测唤醒
                    if interrupt_check_callback():
                        pygame.mixer.music.stop()
                        return True
                else:
                    # 如果没有检测回调，才需要手动 sleep 避免 CPU 100%
                    pygame.time.Clock().tick(30)
            
            try: pygame.mixer.music.unload()
            except: pass
            return False

        except Exception as e:
            print(f"[Voice Error]: {e}", flush=True)
            return False

    def listen(self):
        if not self.vad_model:
            print("[Error] VAD model not loaded.")
            return ""

        # VAD & Audio Config
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 512
        
        PAUSE_THRESHOLD = 0.8 # 静音阈值 (秒)
        PRE_RECORD_SECONDS = 0.3 # 前摇缓冲 (秒)
        MAX_RECORD_SECONDS = 30 # 最大录音时长
        
        pa = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        
        print("\n[Alice]: 聆听中... (请吩咐)", flush=True)
        
        frames = []
        # Ring Buffer 用于保存触发前的声音 (避免丢字)
        pre_record_buffer = deque(maxlen=int(RATE / CHUNK * PRE_RECORD_SECONDS))
        
        triggered = False
        silence_counter = 0
        silence_threshold_chunks = int(PAUSE_THRESHOLD * (RATE / CHUNK))
        max_chunks = int(MAX_RECORD_SECONDS * (RATE / CHUNK))
        
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # 1. 转换音频格式用于 VAD (Int16 -> Float32, Normalized)
                # 修复编译器报错：使用 np.divide 替代 / 运算符
                audio_int16 = np.frombuffer(data, dtype=np.int16)
                audio_float32 = np.divide(audio_int16.astype(np.float32), 32768.0)
                tensor = torch.from_numpy(audio_float32)
                
                # 2. 获取人声概率
                # Silero VAD forward: (x, sr) -> prob
                speech_prob = cast(Any, self.vad_model)(tensor, RATE).item()
                
                if not triggered:
                    pre_record_buffer.append(data)
                    if speech_prob > 0.5:
                        print("[系统]: 检测到人声，开始录音...", flush=True)
                        triggered = True
                        # 将缓冲区的声音写入
                        frames.extend(pre_record_buffer)
                        frames.append(data)
                else:
                    frames.append(data)
                    # 3. 静音检测
                    if speech_prob < 0.5:
                        silence_counter += 1
                    else:
                        silence_counter = 0
                        
                    # 超过静音阈值，停止
                    if silence_counter > silence_threshold_chunks:
                        print("[系统]: 说话结束。", flush=True)
                        break
                    
                    # 超过最大时长，停止
                    if len(frames) > max_chunks:
                        print("[系统]: 达到最大录音时长。", flush=True)
                        break
                        
        except KeyboardInterrupt:
            return ""
        except Exception as e:
            print(f"[Listen Error]: {e}")
            return ""
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            
        if not frames:
            return ""
            
        # 4. 保存为 WAV 并转录
        fname = ""
        text = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wf = wave.open(tmp.name, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pa.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()
                fname = tmp.name
            
            print("[系统]: 正在转录...", flush=True)
            if self.stt_model:
                segments, info = self.stt_model.transcribe(fname, language="zh", beam_size=5)
                # segments 是一个生成器，需要遍历拼接
                text = "".join([segment.text for segment in segments])
            else:
                print("[Error] STT model not initialized.")
        except Exception as e:
            print(f"[Transcribe Error]: {e}")
        finally:
            if fname and os.path.exists(fname):
                try: os.unlink(fname)
                except: pass
                
        return text

    def close(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            pygame.quit() # 彻底退出
        except: pass
        
        if os.path.exists(self.output_file):
            try: os.remove(self.output_file)
            except: pass

class WakeWord:
    def __init__(self):
        try:
            sens = getattr(Config, 'WAKE_SENSITIVITY', 0.7)
            if Config.USE_BUILTIN_KEYWORD:
                self.porcupine = pvporcupine.create(access_key=Config.PICOVOICE_ACCESS_KEY, keywords=['jarvis'], sensitivities=[sens])
            else:
                self.porcupine = pvporcupine.create(access_key=Config.PICOVOICE_ACCESS_KEY, keyword_paths=[Config.WAKE_WORD_FILE], sensitivities=[sens])
        except Exception as e:
            print(f"[WakeWord Init Error]: {e}")
            self.porcupine = None
        
        self.pa = None
        self.stream = None

    def start(self):
        """开启麦克风流"""
        if not self.porcupine: return False
        if self.stream: return True # 已经开启

        try:
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
            return True
        except Exception as e:
            print(f"[WakeWord Start Error]: {e}")
            return False

    def stop(self):
        """关闭麦克风流"""
        if self.stream:
            try: self.stream.close()
            except: pass
            self.stream = None
        if self.pa:
            try: self.pa.terminate()
            except: pass
            self.pa = None

    def process_frame(self):
        """处理一帧音频，检测唤醒词"""
        if not self.stream or not self.porcupine: return False
        try:
            # 这里的 exception_on_overflow=False 是必须的
            # 因为播放音频时 CPU 负载波动，麦克风缓存很容易满，不忽略会报错
            pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            
            pcm_tuple = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            result = self.porcupine.process(pcm_tuple)
            return result >= 0
        except IOError as e:
            # print(f"[Mic Overflow]: {e}") # 调试时可打开，平时忽略
            return False
        except Exception as e:
            print(f"[WakeWord Process Error]: {e}")
            return False

    def listen(self):
        """阻塞式监听 (兼容旧接口)"""
        if not self.start(): return False
        
        print(f"\n[系统]: 待机中...", flush=True)
        try:
            while True:
                if self.process_frame():
                    print("\n>>> 唤醒词触发 <<<", flush=True)
                    return True
        except KeyboardInterrupt: return False
        finally:
            self.stop()
            
    def close(self):
        self.stop()
        if self.porcupine: self.porcupine.delete()

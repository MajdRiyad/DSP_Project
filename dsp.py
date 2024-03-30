import tkinter as tk
import wave
from scipy.fft import fft, fftfreq
import pyaudio
from pathlib import Path
import scipy
from tkinter import Tk, Canvas, Entry,  Button, PhotoImage, filedialog, END
import numpy as np

from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(
    r"images")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("706x386")
window.configure(bg="#AEFFFF")

canvas = Canvas(
    window,
    bg="#AEFFFF",
    height=386,
    width=706,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

canvas.place(x=0, y=0)
canvas.create_rectangle(
    0.0,
    0.0,
    208.0,
    386.0,
    fill="#7DA8B6",
    outline="")

welcome_img = PhotoImage(
    file=relative_to_assets("welcome_img.png"))
image_1 = canvas.create_image(
    451.0,
    61.0,
    image=welcome_img
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
encoding_button = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: encoding_on_click(),
    relief="flat"
)
encoding_button.place(
    x=22.0,
    y=76.0,
    width=164.0,
    height=49.0
)
def encoding_on_click():
    canvas.pack_forget()
    canvas2.pack()
    input_entry.place(
        x=185.0,
        y=99.0,
        width=265.0,
        height=36.0
    )
    submit_btn.place(
        x=235.0,
        y=142.0,
        width=205.0,
        height=34.0

    )
    sav_btn.place(
        x=224.0,
        y=205.0,
        width=205.0,
        height=55.0
    )
    generate_btn.place(
        x=224.0,
        y=281.0,
        width=187.0,
        height=53.0
    )
    back_btn2.place(
    x=520.0,
    y=6.0,
    width=150.0,
    height=30.0
    )




button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
decoding_button = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: decoding_on_click(),
    relief="flat"
)
decoding_button.place(
    x=22.0,
    y=179.0,
    width=164.0,
    height=49.0
)

canvas2 = Canvas(
    window,
    bg="#95C2D0",
    height=386,
    width=706,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)

image_image_2 = PhotoImage(
    file=relative_to_assets("image_2.png"))
image_2 = canvas2.create_image(
    318.0,
    46.0,
    image=image_image_2
)
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_1 = canvas2.create_image(
    317.5,
    118.0,
    image=entry_image_1
)
def create_dictionary ():
    characters = ['a', 'b', 'c', 'd','e','f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    lowFrequencies = [100, 100, 100, 100, 100, 100, 100, 100, 100, 300, 300, 300, 300, 300, 300, 300, 300, 300, 500, 500, 500, 500, 500, 500, 500, 500, 500]
    midFrequencies = [1100, 1100, 1100, 1300, 1300, 1300, 1500, 1500, 1500, 1100, 1100, 1100, 1300, 1300, 1300, 1500, 1500, 1500, 1100, 1100, 1100, 1300, 1300, 1300, 1500, 1500, 1500]
    highFrequencies = [2500, 3000, 3500, 2500, 3000, 3500, 2500, 3000, 3500, 2500, 3000, 3500, 2500, 3000, 3500, 2500, 3000, 3500, 2500, 3000, 3500, 2500, 3000, 3500, 2500, 3000, 3500]
    characterFreq = list(zip(characters,lowFrequencies,midFrequencies,highFrequencies))
    return characterFreq
def encoding (inputString, characterFreq, duration=(40e-3), fs=8000):
    encoded_signal_list = []
    for char in inputString:
        alphabetAndFreq= next((entry for entry in characterFreq if entry[0] == char), None)
        if alphabetAndFreq:
            frequencies = alphabetAndFreq[1:4]
            time = np.linspace(0, duration, int(fs*duration), endpoint=False)
            signal = sum(np.cos( 2 * np.pi * f * time) for f in frequencies)
            encoded_signal_list.append(signal)
    encoded_signal = np.concatenate(encoded_signal_list)
    encoded_signal = (encoded_signal.astype(np.int16))
    return encoded_signal

result=tk.StringVar()
input_entry = tk.Entry(canvas2,textvariable = result)
def encoding_result():
    freq_mapping = create_dictionary()
    encoded_signal_final = encoding(result.get(), freq_mapping)
    return encoded_signal_final

def on_submit_click():
    print(result.get())


button_image_3 = PhotoImage(
    file=relative_to_assets("submit_button.png"))
submit_btn = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: on_submit_click(),
    relief="flat"
)

def save_wave(encoded_signal, filename, fs=8000):
     scipy.io.wavfile.write(filename, fs, encoded_signal)
def save_on_click():
    signal = encoding_result()
    save_wave(signal, filename='output.wav')
    print("File Saved")
    generateSound(signal)
    print("Sound Played")

button_image_4 = PhotoImage(
    file=relative_to_assets("sav_wav_button.png"))
sav_btn = Button(
    image=button_image_4,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: save_on_click(),
    relief="flat"
)
button_image_5 = PhotoImage(
    file=relative_to_assets("generate_signal_button.png"))
generate_btn = Button(
    image=button_image_5,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: generate_on_click(),
    relief="flat"
)
def generateSound(signal):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=8000, output=True)
    stream.write(signal.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def plot_signal(signal, fs=8000):
    plt.figure(figsize=(20, 4))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(signal)) / fs, signal)
    plt.title('Encoded Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    frequency_bins = np.fft.fftfreq(len(signal), d=1/fs)
    fft_values = np.fft.fft(signal)
    plt.subplot(2, 1, 2)
    plt.plot(frequency_bins, np.abs(fft_values))
    plt.title('Fourier Transform')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

def generate_on_click():
    signal = encoding_result()
    plot_signal(signal)
canvas3 = Canvas(
    window,
    bg = "#BFF1F1",
    height = 362,
    width = 699,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
button_image_6 = PhotoImage(
    file=relative_to_assets("browse_btn.png"))
browse_btn = Button(
    image=button_image_6,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: browse_on_click(),
    relief="flat"
)
browse_entry_img = PhotoImage(
    file=relative_to_assets("browse_entry.png"))
browse_entry_img = canvas3.create_image(
    384.5,
    32.5,
    image=browse_entry_img
)
browse_entry = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)

button_image_7 = PhotoImage(
    file=relative_to_assets("freq_btn.png"))
freq_btn = Button(
    image=button_image_7,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: freq_on_click(),
    relief="flat"
)
freq_entry_img = PhotoImage(
    file=relative_to_assets("freq_entry.png"))
freq_entry_img = canvas3.create_image(
    530.0,
    239.0,
    image=freq_entry_img
)
freq_entry = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
button_image_8 = PhotoImage(
    file=relative_to_assets("filter_btn.png"))
filter_btn = Button(
    image=button_image_8,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: filter_on_click(),
    relief="flat"
)
filter_entry_img = PhotoImage(
    file=relative_to_assets("filter_entry.png"))
filter_entry_img = canvas3.create_image(
    205.0,
    239.0,
    image=filter_entry_img
)
filter_entry = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)

image_image_3 = PhotoImage(
    file=relative_to_assets("decodingresultimg.png"))
decoding_result_img = canvas3.create_image(
    384.0,
    103.0,
    image=image_image_3
)

button_image_9 = PhotoImage(
    file=relative_to_assets("calculate_accuracy_btn.png"))
accuracy_btn = Button(
    image=button_image_9,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: accuracy_on_click(),
    relief="flat"
)
calculate_flacc_entry_img = PhotoImage(
    file=relative_to_assets("flaccuracy_entry.png"))
calculate_flacc_entry_img = canvas3.create_image(
    218.5,
    334.5,
    image=calculate_flacc_entry_img
)
calculate_flacc_entry = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
calculate_fqacc_entry_img = PhotoImage(
    file=relative_to_assets("fqaccuracy_entry.png"))
calculate_fqacc_entry_img = canvas3.create_image(
    514.0,
    335.5,
    image=calculate_fqacc_entry_img
)
calculate_fqacc_entry = Entry(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
back_btn2 = Button(
    text="Back to Main Page",
    borderwidth=0,
    fg="#000716",
    bg="#D9D9D9",
    highlightthickness=0,
    command=lambda: back_on_click2(),
    relief="flat"
)

back_btn1 = Button(
    text="Back to Main Page",
    borderwidth=0,
    fg="#000716",
    bg="#D9D9D9",
    highlightthickness=0,
    command=lambda: back_on_click1(),
    relief="flat"
)


def decoding_on_click():
    canvas.pack_forget()
    canvas2.pack_forget()
    canvas3.pack()
    browse_btn.place(
    x=25.0,
    y=12.0,
    width=223.0,
    height=41.0
)
    freq_btn.place(
    x=455.0,
    y=142.0,
    width=134.0,
    height=36.0
)
    filter_btn.place(
    x=153.0,
    y=142.0,
    width=134.0,
    height=36.0
)
    accuracy_btn.place(
    x=270.0,
    y=275.0,
    width=200.0,
    height=34.0
)
    browse_entry.place(
    x=265.0,
    y=6.0,
    width=239.0,
    height=51.0
)

    freq_entry.place(
    x=428.0,
    y=221.0,
    width=204.0,
    height=34.0
)
    filter_entry.place(
    x=103.0,
    y=221.0,
    width=204.0,
    height=34.0
)
    calculate_flacc_entry.place(
    x=137.0,
    y=318.0,
    width=163.0,
    height=31.0
)
    calculate_fqacc_entry.place(
    x=429.0,
    y=320.0,
    width=170.0,
    height=29.0
)
    back_btn1.place(
    x=520.0,
    y=6.0,
    width=150.0,
    height=30.0
    )

def browse_on_click():
    filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if filename:
        browse_entry.delete(0, END)
        browse_entry.insert(0, filename)
fs=8000
duration = 0.04
def decode_fft(chunk):
    yf = fft(chunk)
    xf = fftfreq(n, 1/fs)
    ind = xf > 0
    peak_ind = np.argsort(yf[ind])[-3:]
    return xf[ind][peak_ind]
n = int(0.04 * 8000)
frequenciesDictionary = {
    'a': (100, 1100, 2500),
    'b': (100, 1100, 3000),
    'c': (100, 1100, 3500),
    'd': (100, 1300, 2500),
    'e': (100, 1300, 3000),
    'f': (100, 1300, 3500),
    'g': (100, 1500, 2500),
    'h': (100, 1500, 3000),
    'i': (100, 1500, 3500),
    'j': (300, 1100, 2500),
    'k': (300, 1100, 3000),
    'l': (300, 1100, 3500),
    'm': (300, 1300, 2500),
    'n': (300, 1300, 3000),
    'o': (300, 1300, 3500),
    'p': (300, 1500, 2500),
    'q': (300, 1500, 3000),
    'r': (300, 1500, 3500),
    's': (500, 1100, 2500),
    't': (500, 1100, 3000),
    'u': (500, 1100, 3500),
    'v': (500, 1300, 2500),
    'w': (500, 1300, 3000),
    'x': (500, 1300, 3500),
    'y': (500, 1500, 2500),
    'z': (500, 1500, 3000),
    ' ': (500, 1500, 3500)
}
invertedDictionary = {v: k for k, v in frequenciesDictionary.items()}
def decodeFrequenciesChar(frequencies):
    frequencies = sorted(frequencies)
    frequencies = [int(round(f)) for f in frequencies]
    frequencies = tuple(frequencies)
    if frequencies in invertedDictionary:
        return invertedDictionary[frequencies]
    else:
        return " "
def decodingSignalFFT(signal):
    Segments = np.array_split(signal, len(signal) // n)
    message = ""
    for segment in Segments:
        frequencies = decode_fft(segment)
        char = decodeFrequenciesChar(frequencies)
        message += char
    return message

def freq_on_click():
    filename = browse_entry.get()
    wf = wave.open(filename, 'rb')
    signal = wf.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int16)
    wf.close()
    message = decodingSignalFFT(signal)
    freq_entry.delete(0, END)
    freq_entry.insert(END, message)


def readWavFile(file_path):
    with wave.open(file_path, 'rb') as wave_file:
        audio_data = wave_file.readframes(-1)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sample_rate = wave_file.getframerate()
    return audio_array, sample_rate
def createBandpassFilter(lowCOF, highCOF, fs, order=4):
    nyquist = 0.5 * fs
    low = lowCOF / nyquist
    high = highCOF / nyquist
    numerator, denominator = butter(order, [low, high], btype='band')
    return numerator, denominator
def applyBandpassFilters(segment, numeratorLow, denominatorLow, numeratorMid, denominatorMid, numeratorHigh, denominatorHigh):
    filteredLow = lfilter(numeratorLow, denominatorLow, segment)
    filteredMid = lfilter(numeratorMid, denominatorMid, segment)
    filteredHigh = lfilter(numeratorHigh, denominatorHigh, segment)
    return filteredLow, filteredMid, filteredHigh
def decodeSegment(segment, characterFilters):
    correlationScore = {}

    for char, filters in characterFilters.items():
        numeratorLow, denominatorLow = filters['low']
        numeratorMid, denominatorMid = filters['mid']
        numeratorHigh, denominatorHigh = filters['high']
        filteredLow, filteredMid, filteredHigh = applyBandpassFilters(segment, numeratorLow, denominatorLow, numeratorMid, denominatorMid, numeratorHigh, denominatorHigh)
        frequency_low = frequenciesDictionary[char][0]
        frequency_mid = frequenciesDictionary[char][1]
        frequency_high = frequenciesDictionary[char][2]

        time = np.linspace(0,duration , len(segment), endpoint=False)
        correlationScoreLow = np.max(np.abs(np.correlate(filteredLow, np.cos(2 * np.pi * frequency_low * time), mode='full')))
        correlationScoreMid = np.max(np.abs(np.correlate(filteredMid, np.cos(2 * np.pi * frequency_mid * time), mode='full')))
        correlationScoreHigh = np.max(np.abs(np.correlate(filteredHigh, np.cos(2 * np.pi * frequency_high * time), mode='full')))

        totalCorrelationScore = correlationScoreLow + correlationScoreMid + correlationScoreHigh
        correlationScore[char] = totalCorrelationScore

    return correlationScore
characterFilters = {}
for char, frequencies in frequenciesDictionary.items():
    lowCOF, highCOF = frequencies[0] - 10, frequencies[0] + 10
    numeratorLow, denominatorLow = createBandpassFilter(lowCOF, highCOF, fs)
    lowCOF, highCOF = frequencies[1] - 10, frequencies[1] + 10
    numeratorMid, denominatorMid = createBandpassFilter(lowCOF, highCOF, fs)
    lowCOF, highCOF = frequencies[2] - 10, frequencies[2] + 10
    numeratorHigh, denominatorHigh = createBandpassFilter(lowCOF, highCOF, fs)
    characterFilters[char] = {'low': (numeratorLow, denominatorLow), 'mid': (numeratorMid, denominatorMid), 'high': (numeratorHigh, denominatorHigh)}
def filter_on_click():
    file_path = browse_entry.get()
    audioArray, _ = readWavFile(file_path)
    segmentSize = int(fs * duration)
    stepSize = int(segmentSize * 1)
    segments = [audioArray[i:i + segmentSize] for i in range(0, len(audioArray), stepSize) if
            len(audioArray[i:i + segmentSize]) == segmentSize]
    decodedText = ""
    for segment in segments:
        segmentCorrelationScore = decodeSegment(segment, characterFilters)
        decodedChar = max(segmentCorrelationScore, key=segmentCorrelationScore.get)
        decodedText += decodedChar
    filter_entry.delete(0,END)
    filter_entry.insert(END,decodedText)
def Accuracy(inputText, decodedText):
    matchCount = 0
    minimumLen = min(len(inputText), len(decodedText))

    for char1, char2 in zip(inputText[:minimumLen], decodedText[:minimumLen]):
        if char1 == char2:
            matchCount += 1
    accuracy=(matchCount / len(inputText)) * 100 if len(inputText) > 0 else 0
    string="Accuracy:"+str(accuracy)+"%"
    return string
def accuracy_on_click():
    filter_acc=Accuracy(result.get(), filter_entry.get())
    calculate_flacc_entry.delete(0, END)
    calculate_flacc_entry.insert(END, filter_acc)
    freq_acc=Accuracy(result.get(), freq_entry.get())
    calculate_fqacc_entry.delete(0, END)
    calculate_fqacc_entry.insert(END, freq_acc)

def back_on_click1():
    browse_btn.place_forget()
    browse_entry.place_forget()
    freq_btn.place_forget()
    filter_btn.place_forget()
    filter_entry.place_forget()
    accuracy_btn.place_forget()
    calculate_flacc_entry.place_forget()
    calculate_fqacc_entry.place_forget()
    back_btn1.place_forget()
    freq_entry.place_forget()
    canvas3.pack_forget()
    canvas.pack()

def back_on_click2():
    canvas2.pack_forget()
    canvas.pack()
    submit_btn.place_forget()
    generate_btn.place_forget()
    sav_btn.place_forget()
    back_btn2.place_forget()




window.resizable(False, False)
window.mainloop()

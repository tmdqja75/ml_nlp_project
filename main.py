from __future__ import division

import re
import sys
import os

from google.cloud import speech

import pyaudio
from six.moves import queue
import pickle
import warnings

import time
import pandas as pd

import type_sh.seokho as po
import type_sb.sentence_type as sent_type
import tense_yeram.fin_tense as fin
import type_hw.sentence_classify as cls
import pandas as pd

new_sentence = []
warnings.filterwarnings("ignore")

#환경변수 추가
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('../speechtotext-376312-2323ab2b0356.json')


# Audio recording parameters
RATE = 20000
CHUNK = int(RATE / 10)  # 100ms

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
 In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript


        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one,we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            new_sentence.append(transcript + overwrite_chars)

            return new_sentence
            
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(종료|멈춤)\b", transcript, re.I):
                print("Exiting..")
                
                return transcript
                
                break
                

            num_chars_printed = 0
        
def nlp_project(sentence):
    train = pd.read_csv('./train.csv')
    sentence = ''.join(sentence)

    #################          유형          ########################
    # 유형 분류 모델 불러오기
    type_predict = sent_type.type_ml_model(sentence)
    type_predict = type_predict[0]

    ##################         극성          ########################
    po_model = po.getModel(train)
    to_po_sentence = po.transformTarget(sentence, train)
    # 0 긍정, 1 미정, 2 부정
    po_pred = po_model.predict(to_po_sentence)
    po_pred = "긍정" if po_pred[0] == 0 else "미정" if po_pred[0] == 1 else "부정"

    #################         시제          ########################
    # result = '과거', '현재', '미래'
    tense_pred = fin.tense_ml(sentence)


    ##################         확실성           ########################
    model = pickle.load(open('./type_hy/hy_model.pkl', 'rb'))
    vectorizer = pickle.load(open('./type_hy/hy_vectorizer.pkl', 'rb'))

    sentence_pre = cls.apply_re(cls.last_two_word(sentence))
    sentence_vec = vectorizer.transform([sentence_pre])
    result_classify = cls.predict(sentence_vec)
    if result_classify == 0:
        return "불확실"
    elif result_classify == 1:
        return "확실"
    
    return type_predict, po_pred, tense_pred ,result_classify
    

def custom_tokenizer(sentence):
    from konlpy.tag import Mecab
    '''
    각 문장을 Mecab을 이용하여 토큰화해줄 함수
    토큰들을 리스트 형식으로 반환
    '''
    t= Mecab()
    return [token[0] for token in t.pos(sentence)]
                

def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "ko-KR"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses)  # return new_sentence

    type_pred, po_pred, tense_pred, cls_pred = nlp_project(new_sentence)

    print("문장: ", new_sentence)
    print("문장 유형: ", type_pred)
    print("문장 극성: ", po_pred)
    print("문장 시제: ", tense_pred)
    print("문장 확실성: ", cls_pred)
    print("문장 label: ",type_pred,'-',po_pred,'-',tense_pred,'-',cls_pred)

        

if __name__ == "__main__":

    main()

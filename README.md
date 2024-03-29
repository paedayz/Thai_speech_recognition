# Thai Speech Recognition

โปรแกรมนี้ได้ใช้ model [wav2vec2-large-xlsr-53-th](https://medium.com/airesearch-in-th/airesearch-in-th-3c1019a99cd) เป็นหลัก ในการแปลงข้อมูลเสียงเป็นข้อความ และมีการใช้ [google speech recognition ](https://pypi.org/project/SpeechRecognition/) เป็นตัวเลือกให้ผู้ใช้งานได้เปรียบเทียบประสิทธิภาพ โดยที่ผู้พัฒนานำ `wav2vec2-large-xlsr-53-th` มาเป็น model หลักในการแปลงเสียงเป็นข้อความ เพราะว่า model ตัวนี้สามารถเปลี่ยนคำภาษาไทยเป็นข้อความได้ค่อนข้างแม่นยำ อีกทั้งถ้าหากมีคำภาษาอังกฤษปนอยู่ model ก็สามารถแปลงคำภาษาอังกฤษออกมาเป็นข้อความภาษาไทยได้ ในขนณะที่ `google speech recognition` เมื่อเราเลือกภาษาที่จะแปลงเป็นภาษาไทย (language="th") เมื่อมีคำภาษาอังกฤษปนอยู่ model ตัวนี้จะทำการตัดคำเหล่านั้นออก ทำให้ข้อความขาดหาย หรือผิดเพี้ยนไป (ผลจากการทดสอบของผู้พัฒนา)

<br/>
<br/>

# Generate Word Frequency Report (Optional)
หลังจากที่ได้ข้อความมาจากฟังก์ชัน Speech Recognition ผู้ใช้งานสามารถสร้าง Word frequency report ได้จากฟังก์ชัน `generate_word_frequency_report()` โดยในฟังก์ชันนี้ จะมีการแบ่งคำโดยใช้ [pythainlp 2.0](https://pythainlp.github.io/docs/2.0/api/tokenize.html) (ที่ใช้ version 2.0 เพราะจะสามารถ custom tokenize dict ได้) หลังจากนั้นก็จำทำการนับคำ และนำจำนวนคำที่นับทั้งหมด บันทึกลง report.csv ด้วย [pandas 1.1.5](https://pypi.org/project/pandas/1.1.5/) หากไม่ต้องการสร้าง report ผู้ใช้งานสามารถลบฟังก์ชั่น `generate_word_frequency_report(audio_to_text)` ในไฟล์ app.py ออกได้

<br/>
<br/>

# Seperate voice .wav file (Optional)
เนื่องจากการที่ไฟล์เสียงที่ใหญ่เกินไป อาจจะทำให้ model ทำงานผิดพลาด เพราะฉะนั้นจึงจำเป็นที่จะต้องแบ่งไฟล์เสียงเป็นหลาย ๆ ไฟล์ ทางผู้พัฒนาจึงสร้าง app ที่จะคอยแบ่งไฟล์เสียงของผู้ใช้งานออกเป็นหลาย ๆ ไฟล์ โดยสามารถกำหนดเวลาในหน่วยนาทีของแต่ละไฟล์ได้ ด้วยการรันคำสั่ง
```
>> python seperate_voice.py --min_per_split 1
```
ถ้าหากไม่ใส่ `--min_per_split` แอพจะทำการแยกไฟล์เสียงเป็นไฟล์เสียงละ 2 นาทีโดยอัตโนมัติ

<br/>
<br/>
<br/>

# Installation
## python
```
>> python --version
Python 3.6.8
```
## library
```
>> pip install requirements.txt
```
<br/>
<br/>

# Run application
## `wav2vec2-large-xlsr-53-th` for speech recognition
```
>> python app.py
```

## `google recognition` for speech recognition
```
>> python app.py --recog google
```
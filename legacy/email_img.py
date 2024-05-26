import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import cv2
import numpy as np

# 设置发件人、收件人和邮箱服务器
from_email = '1336458913@qq.com'
to_email = '1336458913@qq.com'
smtp_server = 'smtp.qq.com'
smtp_port = 587  # SMTP 端口号 465 or 587
smtp_username = from_email
smtp_password = os.environ.get('smtp_qq_key')

# 创建邮件对象
msg = MIMEMultipart()
msg['From'] = from_email
msg['To'] = to_email
msg['Subject'] = 'Subject of the email'

target_width = 240
html = '''<html>
            <head>
            <style>
            /* 设置图片容器的样式 */
            .image-container {
                    display: flex; /* 使用 flex 布局 */
                    flex-wrap: wrap; /* 自动换行 */
                    justify-content: center; /* 水平居中 */
            }
  
            /* 设置每张图片的样式 */
                    .image-container img {
                    width: [target_width]px; /* 设置图片宽度 */
                    height: auto; /* 高度自适应 */
                    margin: 5px; /* 图片间距 */
            }
            </style>
            </head>
            <body>
            <div class="image-container">'''.replace('[target_width]', str(target_width))

image_path = r'E:\desktop\yolo\output\388634693999900'

pic_inline = ''
for idx, file in enumerate(os.listdir(image_path)):
    # 读取图片
    img = cv2.imdecode(np.fromfile(f'{image_path}/{file}', dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    ratio = target_width / width
    resized_img = cv2.resize(img, (int(width * ratio), int(height * ratio)))
    _, img_data = cv2.imencode('.jpg', resized_img)

    # 添加图片附件
    image = MIMEImage(img_data.tobytes())
    image.add_header('Content-ID', f'<image{idx}.jpg>')
    msg.attach(image)
    tmp_pic_inline = f'<div><p>{file}</p><img src="cid:image{idx}.jpg" alt="image{idx}.jpg"></div>'
    pic_inline += tmp_pic_inline

html = html + pic_inline + '</div></body></html>'
content = MIMEText(html, 'html', 'utf-8')
msg.attach(content)

# 发送邮件
with smtplib.SMTP(smtp_server, smtp_port) as server:
    server.starttls()
    server.login(smtp_username, smtp_password)
    server.sendmail(from_email, to_email, msg.as_string())
    print("Email sent successfully!")

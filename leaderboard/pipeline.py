import sys
import datetime
from leaderboard.SBT.GA_search import search_based_testing
from drive_upload import compress_and_upload

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart   # ✅ 新增
from email.mime.application import MIMEApplication  # ✅ 新增
from email.utils import formataddr
from email.header import Header


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINE_CONFIG = PROJECT_ROOT / "machine.conf"
MACHINE = MACHINE_CONFIG.read_text(encoding="utf-8").strip()
MACHINE = str(MACHINE).rjust(2,'0')


def perform(setting,agent,line,modules):
    print("Setting: {}, Agent: {}, Line: {}, Modules: {}".format(setting, agent, line, str(modules)))

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")

    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = OUTPUT_DIR / (
        f"output-{formatted_datetime}-{agent}-{line}-{setting}-{str(modules)}.txt"
    )
    print(filename)
    # filename = "/home/guannan/Projects/TCP-Interfuser/output-{}-{}-{}-{}.txt".format(formatted_datetime,agent,line,setting)

    f = open(filename, "w", buffering=1, encoding="utf-8")  # 行缓冲模式
    sys.stdout = f

    search_based_testing(setting, agent, line, modules)

    f.close()

    file = str(filename)
    sender = 'guannanlou@foxmail.com'
    receiver = '492678502@qq.com'
    password = 'mnyfxuortepjbfdd'
    subject = '{}-{}-{}-{}-{}试验结束'.format(MACHINE,formatted_datetime, agent, line, setting)
    content = '试验已结束，请查收。'
    send_qq_email(sender, receiver, password, subject, content, file_path=file)


    local_folder = './data'
    archive_name = "experiment_results_machine_{}".format(MACHINE)
    machine_index= MACHINE
    compress_and_upload(local_folder, archive_name, machine_index)   

    local_folder = './outputs'
    archive_name = "logs_machine_{}".format(MACHINE)
    machine_index= MACHINE
    compress_and_upload(local_folder, archive_name, machine_index) 

def send_qq_email(sender, receiver, password, subject, content, file_path=None):
    # ✅ 改成多部分邮件
    msg = MIMEMultipart()

    msg["From"] = formataddr(("Python程序", sender))
    msg["To"] = receiver
    msg["Subject"] = subject

    # ✅ 邮件正文
    body = MIMEText(content, "plain", "utf-8")
    msg.attach(body)

    # ✅ 如果有附件就添加
    if file_path:
        try:
            with open(file_path, "rb") as f:
                attachment = MIMEApplication(f.read())

                # 处理文件名（防止中文乱码）
                filename = file_path.split("/")[-1]

                attachment.add_header(
                    'Content-Disposition',
                    'attachment',
                    filename=Header(filename, 'utf-8').encode()
                )
                msg.attach(attachment)
        except Exception as e:
            print("附件读取失败:", e)
            return

    try:
        smtp = smtplib.SMTP_SSL("smtp.qq.com", 465)
        smtp.login(sender, password)
        smtp.sendmail(sender, [receiver], msg.as_string())
        smtp.quit()
        print("邮件发送成功！")
    except Exception as e:
        print("邮件发送失败:", e)



# 2 yue - check effect of similarity with unique failure
# perform('GA',         'InterFuser', 'Curve',      ['similarity', 'givenpopulation'])
# perform('GA',           'InterFuser', 'Curve',    ['similarity', 'initpopulation'])
# perform('GBGA',         'InterFuser', 'Curve',    ['similarity', 'initpopulation'])
# perform('GBGA',         'InterFuser', 'Curve',    ['similarity', 'givenpopulation'])


# local similarity 02-09-16:33
# perform('GBGA',         'InterFuser', 'Curve',    ['local_similarity', 'initpopulation'])
# perform('GA',           'InterFuser', 'Curve',    ['local_similarity', 'initpopulation'])

# perform('smartrandom',  'InterFuser', 'Curve',    ['initpopulation'])
# perform('GBGA',         'InterFuser', 'Curve',    ['similarity', 'initpopulation'])
# perform('GA',           'InterFuser', 'Curve',    ['similarity', 'initpopulation'])


# collision similarity 02-13-15:00
# perform('GBGA',         'InterFuser', 'Curve',    ['has_collision_similarity', 'initpopulation'])
# perform('GA',           'InterFuser', 'Curve',    ['has_collision_similarity', 'initpopulation'])



# random 3-6-18:49
# perform('smartrandom',  'InterFuser', 'Curve',    ['initpopulation'])
# perform('smartrandom',  'InterFuser', 'Curve',    ['initpopulation'])



# GA 3-6-18:49
# use collision feature and similarity to guide GA search, without increase runs of surrogate

# perform('GA',         'InterFuser', 'Curve',    ['similarity', 'collision_similarity', 'initpopulation'])
# perform('GBGA',         'InterFuser', 'Curve',    ['similarity', 'collision_similarity', 'initpopulation'])



# perform('GBGA',         'InterFuser', 'Curve',    ['initpopulation'])
# perform('GBGA',         'InterFuser', 'Curve',    ['similarity', 'initpopulation'])
# perform('GBGA',         'InterFuser', 'Curve',    ['collision_similarity', 'initpopulation'])
# perform('GBGA',         'InterFuser', 'Curve',    ['similarity', 'collision_similarity', 'initpopulation'])

# perform('GA',           'InterFuser', 'Curve',    ['similarity', 'collision_similarity', 'initpopulation'])
# perform('smartrandom',  'InterFuser', 'Curve',    ['initpopulation'])
# perform('random',       'InterFuser', 'curve',    ['initpopulation'])

# perform('GA',           'InterFuser', 'Straight', ['similarity', 'collision_similarity', 'initpopulation'])
# perform('GBGA',         'InterFuser', 'Straight', ['similarity', 'collision_similarity', 'initpopulation'])
# perform('random',       'InterFuser', 'Straight', ['initpopulation'])
# perform('smartrandom',  'InterFuser', 'Straight', ['initpopulation'])


# perform('GA',           'InterFuser', 'Curve', ['initpopulation'])
# perform('GA',           'InterFuser', 'Curve', ['initpopulation', 'similarity'])
# perform('GA',           'InterFuser', 'Curve', ['initpopulation', 'collision_similarity'])
# perform('GA',           'InterFuser', 'Curve', ['initpopulation', 'similarity', 'collision_similarity'])

# perform('smartrandom',  'InterFuser', 'Curve',    ['initpopulation'])

print("Experiments Start")


perform('GA',           'InterFuser', 'Curve', ['initpopulation'])
perform('GA',           'InterFuser', 'Curve', ['initpopulation', 'similarity'])
perform('GA',           'InterFuser', 'Curve', ['initpopulation', 'collision_similarity'])
perform('GA',           'InterFuser', 'Curve', ['initpopulation', 'similarity', 'collision_similarity'])

sender = 'guannanlou@foxmail.com'
receiver = '492678502@qq.com'
password = 'mnyfxuortepjbfdd'
subject = '试验结束'
content = '试验已结束，请查收。'

send_qq_email(sender, receiver, password, subject, content, file_path=None)

import sys, json, socket, datetime, smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import formataddr
from email.header import Header
from leaderboard.SBT.GA_search import search_based_testing
from drive_upload import compress_and_upload

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "machine.conf"
HOST = socket.gethostname()
SENDER, RECEIVER, PASSWORD = "guannanlou@foxmail.com", "492678502@qq.com", "mnyfxuortepjbfdd"

def load_machine_id():
    with CONFIG.open("r", encoding="utf-8") as f: config = json.load(f)
    if HOST not in config: raise KeyError(f"{HOST!r} not found in {CONFIG}")
    return str(config[HOST]).zfill(3)

MACHINE = load_machine_id()

def get_experiment_name(setting, agent, line, modules):
    s, c = "similarity" in modules, "collision_similarity" in modules
    fitness = "Both" if s and c else "ScenarioSimilarity" if s else "CollisionSimilarity" if c else "Original"
    return f"{setting}-{agent}-{line}", fitness

def send_qq_email(subject, content="试验已结束，请查收。", file_path=None):
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = formataddr(("Python程序", SENDER)), RECEIVER, subject
    msg.attach(MIMEText(content, "plain", "utf-8"))

    if file_path:
        try:
            with open(file_path, "rb") as f: attachment = MIMEApplication(f.read())
            attachment.add_header("Content-Disposition", "attachment", filename=Header(Path(file_path).name, "utf-8").encode())
            msg.attach(attachment)
        except Exception as e: print("附件读取失败:", e)

    try:
        with smtplib.SMTP_SSL("smtp.qq.com", 465) as smtp:
            smtp.login(SENDER, PASSWORD); smtp.sendmail(SENDER, [RECEIVER], msg.as_string())
        print("邮件发送成功！")
    except Exception as e: print("邮件发送失败:", e)

def perform(setting, agent, line, modules):
    group, fitness = get_experiment_name(setting, agent, line, modules)
    remote = f"{group}/{fitness}"
    now = datetime.datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    out_dir = ROOT / "outputs"; out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"output-{now}-{agent}-{line}-{setting}-{modules}.txt"

    print(f"Setting: {setting}, Agent: {agent}, Line: {line}, Modules: {modules}")
    print(f"Remote: machine_{MACHINE}/{remote}")
    print(filename)

    stdout = sys.stdout
    try:
        with filename.open("w", buffering=1, encoding="utf-8") as f:
            sys.stdout = f
            search_based_testing(setting, agent, line, modules)
    finally:
        sys.stdout = stdout

    send_qq_email(f"{MACHINE}-{now}-{agent}-{line}-{setting}试验结束", file_path=str(filename))
    compress_and_upload("./data", f"experiment_results_machine_{MACHINE}", MACHINE, remote_subfolder=remote)
    compress_and_upload("./outputs", f"logs_machine_{MACHINE}", MACHINE, remote_subfolder=remote)


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


perform('GA',           'InterFuser', 'Straight', ['initpopulation'])
perform('GA',           'InterFuser', 'Straight', ['initpopulation', 'similarity'])
perform('GA',           'InterFuser', 'Straight', ['initpopulation', 'collision_similarity'])
perform('GA',           'InterFuser', 'Straight', ['initpopulation', 'similarity', 'collision_similarity'])

sender = 'guannanlou@foxmail.com'
receiver = '492678502@qq.com'
password = 'mnyfxuortepjbfdd'
subject = '试验结束'
content = '试验已结束，请查收。'

send_qq_email(sender, receiver, password, subject, content, file_path=None)

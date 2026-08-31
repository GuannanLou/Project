import sys, json, socket, datetime, smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.utils import formataddr
from email.header import Header
from leaderboard.SBT.GA_search import search_based_testing
from drive_upload import compress_selected_and_upload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINE_CONFIG_PATH = PROJECT_ROOT / "machine.conf"

def load_machine_id():
    hostname = socket.gethostname()
    if not MACHINE_CONFIG_PATH.exists(): raise FileNotFoundError(f"Machine configuration file not found: {MACHINE_CONFIG_PATH}")
    with MACHINE_CONFIG_PATH.open("r", encoding="utf-8") as file: machine_config = json.load(file)
    if hostname not in machine_config:
        available_hosts = ", ".join(machine_config.keys())
        raise KeyError(f"Hostname {hostname!r} is not configured in {MACHINE_CONFIG_PATH}. Configured hostnames: {available_hosts}")
    return str(machine_config[hostname]).zfill(2)

HOSTNAME = socket.gethostname()
MACHINE = load_machine_id()

def get_experiment_name(setting, agent, line, modules):
    base = f"{setting}-{agent}-{line}"
    has_s, has_c = "similarity" in modules, "collision_similarity" in modules
    fitness = "Both" if has_s and has_c else "ScenarioSimilarity" if has_s else "CollisionSimilarity" if has_c else "Original"
    return base, fitness

def perform(setting,agent,line,modules):
    print("Setting: {}, Agent: {}, Line: {}, Modules: {}".format(setting, agent, line, str(modules)))

    experiment_group, fitness_setting = get_experiment_name(setting, agent, line, modules)
    remote_subfolder = f"{experiment_group}/{fitness_setting}"

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d|%H:%M:%S")

    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = OUTPUT_DIR / f"output-{formatted_datetime}-{agent}-{line}-{setting}-{str(modules)}.txt"
    print(filename)

    data_root = PROJECT_ROOT / "data" / agent
    data_root.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in data_root.iterdir() if p.is_dir()}

    f = open(filename, "w", buffering=1, encoding="utf-8")
    sys.stdout = f
    search_based_testing(setting, agent, line, modules)
    f.close()

    after = {p.name for p in data_root.iterdir() if p.is_dir()}
    new_paths = [data_root / x for x in sorted(after - before)]

    file = str(filename)
    sender = 'guannanlou@foxmail.com'
    receiver = '492678502@qq.com'
    password = 'mnyfxuortepjbfdd'
    subject = '{}-{}-{}-{}-{}试验结束'.format(MACHINE,formatted_datetime, agent, line, setting)
    content = '试验已结束，请查收。'
    send_qq_email(sender, receiver, password, subject, content, file_path=file)

    compress_selected_and_upload(new_paths, f"experiment_results_machine_{MACHINE}", MACHINE, remote_subfolder=remote_subfolder)
    compress_selected_and_upload([filename], f"logs_machine_{MACHINE}", MACHINE, remote_subfolder=remote_subfolder)

def send_qq_email(sender, receiver, password, subject, content, file_path=None):
    msg = MIMEMultipart()
    msg["From"] = formataddr(("Python程序", sender))
    msg["To"] = receiver
    msg["Subject"] = subject

    body = MIMEText(content, "plain", "utf-8")
    msg.attach(body)

    if file_path:
        try:
            with open(file_path, "rb") as f: attachment = MIMEApplication(f.read())
            filename = file_path.split("/")[-1]
            attachment.add_header('Content-Disposition','attachment',filename=Header(filename, 'utf-8').encode())
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


# perform('random',       'InterFuser', 'Curve', [''])
# perform('smartrandom',  'InterFuser', 'Curve', ['initpopulation'])
# perform('random',       'InterFuser', 'Straight', [''])
# perform('smartrandom',  'InterFuser', 'Straight', ['initpopulation'])

# perform('GBGA',           'InterFuser', 'Curve', ['initpopulation'])
# perform('GBGA',           'InterFuser', 'Curve', ['initpopulation', 'similarity'])
# perform('GBGA',           'InterFuser', 'Curve', ['initpopulation', 'collision_similarity'])

perform('random',           'TCP', 'Curve', [])
perform('smartrandom',      'TCP', 'Curve', ['initpopulation'])
perform('GBGA',             'TCP', 'Curve', ['initpopulation', 'similarity', 'collision_similarity'])
perform('GA',               'TCP', 'Curve', ['initpopulation', 'similarity', 'collision_similarity'])


perform('smartrandom',      'TCP', 'Straight', ['initpopulation'])
perform('random',           'TCP', 'Straight', [])
perform('GA',               'TCP', 'Straight', ['initpopulation', 'similarity', 'collision_similarity'])
perform('GBGA',             'TCP', 'Straight', ['initpopulation', 'similarity', 'collision_similarity'])



sender = 'guannanlou@foxmail.com'
receiver = '492678502@qq.com'
password = 'mnyfxuortepjbfdd'
subject = '试验结束'
content = '试验已结束，请查收。'

send_qq_email(sender, receiver, password, subject, content, file_path=None)

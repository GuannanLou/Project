import sys, os, json, socket, datetime, smtplib
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

    original_stdout = sys.stdout
    f = open(filename, "w", buffering=1, encoding="utf-8")
    try:
        sys.stdout = f
        search_based_testing(setting, agent, line, modules)
    finally:
        sys.stdout = original_stdout
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
    if not password:
        print("未设置 QQ_EMAIL_AUTH_CODE，跳过邮件通知。")
        return

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

# 按优先级列出尚未完成的实验及需要运行它们的机器。
# 顺序：GBGA-TCP-Straight-Both -> 其他 TCP -> InterFuser。
# MACHINE 由 machine.conf 读取，下面统一使用整数编号，避免 "01" 与 1 不匹配。
PENDING_EXPERIMENTS = [
    ({14}), 'GA',     'TCP', 'Straight', ['initpopulation', 'similarity', 'collision_similarity'],
    ({14}), 'random', 'TCP', 'Straight', ['initpopulation', 'similarity', 'collision_similarity'],
    
    
    # # 1. 所有机器首先运行 GBGA-TCP-Straight-Both
    # (range(1, 16), 'GBGA',        'TCP',        'Straight', ['initpopulation', 'similarity', 'collision_similarity']),

    # # 2. 补齐其余 TCP 实验
    # ({1, 2, 3, 5, 7, 9, 12, 13}, 'GBGA',       'TCP',        'Curve',    ['initpopulation', 'similarity', 'collision_similarity']),
    # ({1, 2, 3, 5, 7, 9, 12, 13}, 'GA',         'TCP',        'Curve',    ['initpopulation', 'similarity', 'collision_similarity']),
    # ({1, 4, 6, 8, 9, 10, 11, 13, 15}, 'GA',    'TCP',        'Straight', ['initpopulation', 'similarity', 'collision_similarity']),
    # ({1, 2, 3, 5, 7, 9, 12, 13}, 'random',     'TCP',        'Curve',    []),
    # ({4, 6, 8, 10, 11, 13, 15},  'random',     'TCP',        'Straight', []),
    # ({1, 2, 3, 5, 7, 9, 12, 13}, 'smartrandom','TCP',        'Curve',    ['initpopulation']),
    # ({4, 8, 11, 15},              'smartrandom','TCP',        'Straight', ['initpopulation']),

    # # 3. 最后补齐 InterFuser 实验
    # ({3, 4, 13},                  'GA',          'InterFuser', 'Curve',    ['initpopulation', 'similarity', 'collision_similarity']),
    # ({3},                         'GA',          'InterFuser', 'Curve',    ['initpopulation', 'collision_similarity']),
    # ({5, 11},                     'GBGA',        'InterFuser', 'Curve',    ['initpopulation', 'similarity', 'collision_similarity']),
]

machine_id = int(MACHINE)
machine_queue = [
    (setting, agent, line, modules)
    for machines, setting, agent, line, modules in PENDING_EXPERIMENTS
    if machine_id in machines
]

print(f"Machine {MACHINE}: {len(machine_queue)} pending experiment(s)")
for index, (setting, agent, line, modules) in enumerate(machine_queue, 1):
    experiment_group, fitness_setting = get_experiment_name(setting, agent, line, modules)
    print(f"[{index}/{len(machine_queue)}] {experiment_group}/{fitness_setting}")
    perform(setting, agent, line, modules)

sender = 'guannanlou@foxmail.com'
receiver = '492678502@qq.com'
password = os.getenv('QQ_EMAIL_AUTH_CODE')
subject = f'{MACHINE}号机器全部补充试验结束'
content = f'{MACHINE}号机器的{len(machine_queue)}项补充试验已全部结束，请查收。'

send_qq_email(sender, receiver, password, subject, content, file_path=None)

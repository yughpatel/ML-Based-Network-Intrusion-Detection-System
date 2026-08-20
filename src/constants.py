# NSL-KDD column schema and label taxonomy.
#
# The raw KDDTrain+.txt / KDDTest+.txt files have no header row. Column 42
# (index 41) is the attack name (string, e.g. "normal", "neptune"); column 43
# (index 42) is a 0-21 "difficulty score" describing how hard the record was
# for the original 21 KDD'99 learners to classify -- it is metadata about the
# record, not a usable feature or a label, and must be dropped before
# training.

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack_type", "difficulty_level",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NON_FEATURE_COLUMNS = ["attack_type", "difficulty_level"]

# Attack-type -> attack-category taxonomy, covering the union of attack types
# present in KDDTrain+ (23, incl. normal) and the official KDDTest+ (38, incl.
# normal), 39 attack types total. KDDTest+ includes attack types never seen in
# training (e.g. apache2, httptunnel, mailbomb, mscan, named, processtable,
# ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm,
# xlock, xsnoop, xterm) -- this is intentional in NSL-KDD, designed to test
# generalization to novel attacks. Grouping is the standard 5-class NSL-KDD
# taxonomy (Normal / DoS / Probe / R2L / U2R) used across the literature
# (e.g. Dhanabal & Shantharajah, 2015).
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos", "processtable": "dos",
    "udpstorm": "dos", "apache2": "dos", "worm": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

RANDOM_STATE = 42

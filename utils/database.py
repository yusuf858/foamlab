"""
FoamLab — Database Layer v4 (Final)
Fixes:
  - MySQL 8 caching_sha2_password support via auth_plugin override
  - connect_timeout=5 always
  - Detailed error stored and returned to UI
  - init_db connects without DB first, creates it, then creates tables
  - check_db_status returns bool cleanly
"""
import logging
logger = logging.getLogger(__name__)

# Auto-load credentials from db_config.py if present
try:
    import db_config as _dbc
    _host = _dbc.DB_HOST
    _port = _dbc.DB_PORT
    _user = _dbc.DB_USER
    _pass = _dbc.DB_PASS
    _name = _dbc.DB_NAME
except Exception:
    _host, _port, _user, _pass, _name = "localhost", 3306, "root", "", "foamlab_db"

DB_CONFIG = {
    "host":            _host,
    "port":            _port,
    "user":            _user,
    "password":        _pass,
    "charset":         "utf8mb4",
    "autocommit":      False,
    "connect_timeout": 5,
}
DB_NAME = _name
_last_error = ""


def _get_conn(with_db=True):
    global _last_error
    try:
        import pymysql
        cfg = dict(DB_CONFIG)
        if with_db:
            cfg["database"] = DB_NAME
        # Try normal connection first
        try:
            conn = pymysql.connect(**cfg)
            _last_error = ""
            return conn
        except pymysql.err.OperationalError as e:
            err_str = str(e)
            # MySQL 8 caching_sha2_password — retry with explicit auth plugin
            if "authentication" in err_str.lower() or "plugin" in err_str.lower() or "2059" in err_str:
                cfg2 = dict(cfg)
                cfg2["auth_plugin"] = "mysql_native_password"
                try:
                    conn = pymysql.connect(**cfg2)
                    _last_error = ""
                    return conn
                except Exception:
                    pass  # fall through to raise original
            raise
    except Exception as e:
        _last_error = str(e)
        logger.debug(f"DB connect failed: {e}")
        return None


def check_db_status():
    conn = _get_conn(with_db=True)
    if conn:
        conn.close()
        return True
    return False


def get_last_error():
    return _last_error


def init_db():
    global _last_error
    # Step 1: connect without specifying a database
    conn = _get_conn(with_db=False)
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cur.execute(f"USE `{DB_NAME}`")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analysis_log (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    filename      VARCHAR(255)  NOT NULL DEFAULT '',
                    filter_type   VARCHAR(32)   NOT NULL DEFAULT '',
                    enhance       VARCHAR(32)   NOT NULL DEFAULT '',
                    seg_method    VARCHAR(32)   NOT NULL DEFAULT '',
                    severity      VARCHAR(32)   NOT NULL DEFAULT 'Clean',
                    dominant      VARCHAR(64)   NOT NULL DEFAULT '',
                    total_regions INT           NOT NULL DEFAULT 0,
                    pollution_pct FLOAT         NOT NULL DEFAULT 0,
                    process_time  FLOAT         NOT NULL DEFAULT 0,
                    location      VARCHAR(255)  NOT NULL DEFAULT '',
                    created_at    DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS region_log (
                    id           INT AUTO_INCREMENT PRIMARY KEY,
                    analysis_id  INT           NOT NULL DEFAULT 0,
                    label        VARCHAR(64)   NOT NULL DEFAULT '',
                    area         DOUBLE        NOT NULL DEFAULT 0,
                    circularity  DOUBLE        NOT NULL DEFAULT 0,
                    compactness  DOUBLE        NOT NULL DEFAULT 0,
                    aspect_ratio DOUBLE        NOT NULL DEFAULT 0,
                    solidity     DOUBLE        NOT NULL DEFAULT 0,
                    pct_image    DOUBLE        NOT NULL DEFAULT 0,
                    created_at   DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
        conn.close()
        logger.info(f"DB '{DB_NAME}' ready")
        return True
    except Exception as e:
        _last_error = str(e)
        logger.error(f"init_db: {e}")
        try: conn.rollback(); conn.close()
        except: pass
        return False


def log_analysis(filename, result, location=""):
    conn = _get_conn(with_db=True)
    if not conn:
        return -1
    try:
        p = result.get("params", {})
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analysis_log
                  (filename,filter_type,enhance,seg_method,severity,dominant,
                   total_regions,pollution_pct,process_time,location)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (str(filename), str(p.get("filter","")), str(p.get("enhance","")),
                  str(p.get("segment","")), str(result.get("severity","Clean")),
                  str(result.get("dominant_label","")), int(result.get("total_regions",0)),
                  float(result.get("pollution_area_pct",0)), float(result.get("process_time",0)),
                  str(location)))
            aid = cur.lastrowid
            for r in result.get("regions",[])[:20]:
                cur.execute("""
                    INSERT INTO region_log
                      (analysis_id,label,area,circularity,compactness,aspect_ratio,solidity,pct_image)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """, (aid, str(r.get("label","")), float(r.get("area",0)),
                      float(r.get("circularity",0)), float(r.get("compactness",0)),
                      float(r.get("aspect_ratio",0)), float(r.get("solidity",0)),
                      float(r.get("pct_image",0))))
        conn.commit()
        conn.close()
        return aid
    except Exception as e:
        logger.error(f"log_analysis: {e}")
        try: conn.rollback(); conn.close()
        except: pass
        return -1


def get_history(limit=200):
    conn = _get_conn(with_db=True)
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id,filename,filter_type,enhance,seg_method,severity,
                       dominant,total_regions,pollution_pct,process_time,location,
                       DATE_FORMAT(created_at,'%%Y-%%m-%%d %%H:%%i') AS ts
                FROM analysis_log ORDER BY created_at DESC LIMIT %s
            """, (limit,))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"get_history: {e}")
        try: conn.close()
        except: pass
        return []


def get_trend_data():
    conn = _get_conn(with_db=True)
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DATE(created_at) AS d,
                       ROUND(AVG(pollution_pct),2) AS avg_poll,
                       COUNT(*) AS cnt
                FROM analysis_log
                GROUP BY DATE(created_at)
                ORDER BY d DESC LIMIT 30
            """)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"get_trend_data: {e}")
        try: conn.close()
        except: pass
        return []
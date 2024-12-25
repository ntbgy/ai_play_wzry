sqlite3 AiPlayWzryDb.db

.show
.header on
.mode column
.timer on

.open AiPlayWzryDb.db

.databases

CREATE TABLE training_data(
   name      CHAR(10)      PRIMARY KEY     NOT NULL,
   root_path VARCHAR(256)                  NOT NULL,
   score     REAL,
   tag       VARCHAR(8)
);

CREATE INDEX name ON training_data(name);

.schema training_data

INSERT INTO training_data (name,root_path,score,tag)
VALUES ('test', 'E:\\ai-play-wzry\\训练数据样本\\未用', 234.93, '未知');

select * from training_data where name = '1726126615';

DELETE FROM training_data WHERE name = '1726126615';

CREATE TABLE training_full_data(
   id            INTEGER PRIMARY KEY AUTOINCREMENT,
   root_path     VARCHAR(256)  NOT NULL,
   name          CHAR(10)      NOT NULL,
   image_name    VARCHAR(256)  NOT NULL,
   action_touch  VARCHAR(256)  NOT NULL,
   state         VARCHAR(256),
   score         REAL,
   create_time   char(16),
   update_time   char(16),
   exist         TEXT DEFAULT 'True',
   -- 创建唯一索引
   UNIQUE(id)
);

INSERT INTO training_full_data
(name, root_path, action_touch, image_name, state, score, create_time, update_time)
VALUES(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
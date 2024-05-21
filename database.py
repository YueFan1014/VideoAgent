import os
import json
import pickle
from collections import defaultdict
from encoder import encode_sentences
from utils import compute_cosine_similarity, top_k_indices
import numpy as np
import sqlite3


class DataBase:
    def __init__(self, video_path, base_dir='preprocess', use_reid=True):
        base_name = os.path.basename(video_path).replace(".mp4", "")
        self.video_dir = os.path.join(base_dir, base_name)
        self.use_reid = use_reid
        if self.use_reid:
            with open(os.path.join(self.video_dir, 'reid.pkl'), 'rb') as f:
                content = pickle.load(f)
            self.frame2uid, self.uid2frame, self.uid2category = content[0], content[1], content[2]
            with open(os.path.join(self.video_dir, 'uid2clip.pkl'), 'rb') as f:
                self.uid2emb = pickle.load(f)
        else:
            with open(os.path.join(self.video_dir, 'tracking.pkl'), 'rb') as f:
                content = pickle.load(f)
            self.frame2uid, self.uid2frame, self.uid2category = content[0], content[1], content[2]
            with open(os.path.join(self.video_dir, 'tid2clip.pkl'), 'rb') as f:
                self.uid2emb = pickle.load(f)

        with open(os.path.join(self.video_dir, 'segment2id.json')) as f:
            self.segment2id = json.load(f)
        self.segment_id2uids = defaultdict(set)
        for frame in self.frame2uid:
            segment_id = 0
            for segment in self.segment2id:
                start, end = segment.split('_')
                start, end = int(start), int(end)
                if start <= frame <= end:
                    segment_id = self.segment2id[segment]
                    break
            uids = list(self.frame2uid[frame])
            self.segment_id2uids[segment_id].update(uids)
        
        if os.path.exists('database.db'):
            os.remove('database.db')
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()
        create_object = """
        CREATE TABLE Objects(
            object_id INT,
            category VARCHAR(255),
            PRIMARY KEY (object_id)
        );
        """
        cursor.execute(create_object)
        create_segment = """
        CREATE TABLE Segments(
            segment_id INT,
            PRIMARY KEY (segment_id)
        );
        """
        cursor.execute(create_segment)
        create_object_segment = """
        CREATE TABLE Objects_Segments(
            object_id INT,
            segment_id INT,
            PRIMARY KEY (object_id, segment_id),
            FOREIGN KEY (object_id) REFERENCES Objects(object_id),
            FOREIGN KEY (segment_id) REFERENCES Segments(segment_id)
        );
        """
        cursor.execute(create_object_segment)
        connection.commit()

        insert_objects = []
        for uid in self.uid2category:
            line = "INSERT INTO Objects (object_id, category) VALUES ({}, '{}')".format(str(uid), self.uid2category[uid])
            #print(line)
            insert_objects.append(line)
        for s in insert_objects:
            cursor.execute(s)

        insert_segments = []
        for segment in self.segment2id:
            segment_id = self.segment2id[segment]
            line = "INSERT INTO Segments (segment_id) VALUES ({})".format(str(segment_id))
            #print(line)
            insert_segments.append(line)
        for s in insert_segments:
            cursor.execute(s)


        insert_object_segments = []
        for segment_id in self.segment_id2uids:
            for uid in self.segment_id2uids[segment_id]:
                line = "INSERT INTO Objects_Segments (object_id, segment_id) VALUES ({}, {})".format(str(uid), str(segment_id))
                #print(line)
                insert_object_segments.append(line)
        for s in insert_object_segments:
            cursor.execute(s)
        
        connection.commit()
        cursor.close()
        connection.close()


    def retrieve_candidate_objects(self, description):
        des_emb = encode_sentences([f"a photo of a {description}."], model_name='clip')
        scores = compute_cosine_similarity(des_emb, list(self.uid2emb.values()))
        indices = np.where(scores >= 0.26)[0]
        candidate_uids = []
        for i in indices:
            candidate_uids.append(list(self.uid2emb)[i])
        return candidate_uids


    def query_database(self, program):
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()
        try:
            cursor.execute(program)
            results = cursor.fetchall()
            return results
        except sqlite3.Error as e:
            return e
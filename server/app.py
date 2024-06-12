from flask import Flask, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import os

app = Flask(__name__, static_folder='static')  # 如果index.html在static目录下
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gps.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class GPSData(db.Model):
    hardware_id = db.Column(db.String(80), primary_key=True)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    altitude = db.Column(db.Float)
    accuracy = db.Column(db.Float)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/send_gps', methods=['POST'])
def receive_gps():
    data = request.json
    existing_data = GPSData.query.filter_by(hardware_id=data['hardwareId']).first()
    if existing_data:
        # 更新现有记录
        existing_data.latitude = data['latitude']
        existing_data.longitude = data['longitude']
        existing_data.altitude = data['altitude']
        existing_data.accuracy = data['accuracy']
    else:
        # 创建新记录
        new_gps_data = GPSData(hardware_id=data['hardwareId'], latitude=data['latitude'], longitude=data['longitude'],
                                altitude=data['altitude'], accuracy=data['accuracy'])
        db.session.add(new_gps_data)
    db.session.commit()
    return 'GPS data received successfully!', 200


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

-- Create the database with UTF8 character set and InnoDB engine for robust transactional support
CREATE DATABASE IF NOT EXISTS student_attendance
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;
USE student_attendance;

-- Create the students table with appropriate constraints and indexes
CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    rfid_card VARCHAR(50) UNIQUE,         -- Allowing NULL for students without RFID
    face_data VARCHAR(100),               -- Could store a reference to face recognition data if needed
    qr_code VARCHAR(255),                 -- Stores either the QR code text or the image file path
    INDEX idx_student_id (student_id)     -- Additional index for performance on joins/searches
) ENGINE=InnoDB;

-- Create the attendance table with a foreign key constraint for referential integrity,
-- default timestamp, and proper indexing for performance.
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(20) NOT NULL,
    method VARCHAR(20) NOT NULL,          -- Marked NOT NULL if every record must include a method
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_student
      FOREIGN KEY (student_id)
      REFERENCES students(student_id)
      ON DELETE CASCADE,
    INDEX idx_attendance_student (student_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB;

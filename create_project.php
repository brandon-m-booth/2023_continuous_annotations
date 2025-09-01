<?php
session_start();

ob_start(); // Fixes "headers already sent" warning

$title = 'Platform for Affective Game ANnotation';
$css = ['researcher.css', 'forms.css'];
include("base.php");

if (!isset($_SESSION["loggedin"]) || $_SESSION["loggedin"] !== true) {
    header("location: login.php");
    exit;
}

require_once "config.php";

$username = $_SESSION["username"];
$project_id = bin2hex(random_bytes(16));

// Safely retrieve POST data with defaults to prevent warnings
$project_name = $_POST["project_name"] ?? '';
$target = $_POST["target"] ?? '';
$type = $_POST["type"] ?? '';
$source_type = $_POST["source_type"] ?? '';
$video_loading = $_POST["video_loading"] ?? '';
$endless = isset($_POST["endless"]) ? 'on' : 'off';
$sound = $_POST["video_sound"] ?? 'on';

$upload_message = htmlspecialchars(trim($_POST["upload-message"] ?? ''), ENT_QUOTES);
$start_message = htmlspecialchars(trim($_POST["start-message"] ?? ''), ENT_QUOTES);
$end_message = htmlspecialchars(trim($_POST["end-message"] ?? ''), ENT_QUOTES);
$survey_link = trim($_POST["survey-link"] ?? '');
$autofill_id = trim($_POST["autofill-id"] ?? '');

$source_urls = $_POST["source_url"] ?? [];
$files = $_FILES['file'] ?? [];

$n_of_entries = 0;
$n_of_participant_runs = 0;
$source_url_err = $project_name_err = $target_err = '';

// Validation
if (empty($project_name)) $project_name_err = "Please enter a project name.";
if (empty($target)) $target_err = "Please enter an annotation target.";
if ($source_type === 'youtube' && count(array_filter($source_urls)) === 0) {
    $source_url_err = "Please enter at least one YouTube URL.";
}
if ($source_type === 'upload' && empty(array_filter($files['name'] ?? []))) {
    $source_url_err = "Please select at least one file to upload.";
}

// Count entries
if ($source_type === 'youtube') {
    $n_of_entries = count(array_filter($source_urls));
} elseif ($source_type === 'upload') {
    $n_of_entries = count(array_filter($files['name'] ?? []));
} elseif (in_array($source_type, ['user_upload', 'user_youtube'])) {
    $n_of_entries = (int)($_POST['n_of_participant_uploads'] ?? 0);
}

// Participant run logic
if ($video_loading === 'random') {
    $n_of_participant_runs = (int)($_POST["n_of_participant_runs"] ?? 0);
    if ($n_of_participant_runs <= 0) {
        $n_of_participant_runs = $n_of_entries;
    }
} else {
    $n_of_participant_runs = $n_of_entries;
}

// If everything validates, insert into DB
if (empty($project_name_err) && empty($target_err) && empty($source_url_err)) {
    $stmt = $pdo->prepare("INSERT INTO projects (
        username, project_id, project_name, target, type, source_type,
        video_loading, endless, n_of_entries, n_of_participant_runs, sound,
        upload_message, start_message, end_message, survey_link, autofill_id, archived
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");

    $stmt->execute([
        $username, $project_id, $project_name, $target, $type, $source_type,
        $video_loading, $endless, $n_of_entries, $n_of_participant_runs, $sound,
        $upload_message, $start_message, $end_message, $survey_link, $autofill_id, 'false'
    ]);

    // Insert entries
    $upload_dir = "user_uploads/";
    for ($i = 0; $i < $n_of_entries; $i++) {
        $entry_id = $i + 1;
        $original_name = $source_url = '';

        if ($source_type === 'youtube') {
            $source_url = trim($source_urls[$i] ?? '');
            $original_name = explode("=", $source_url)[1] ?? "entry{$entry_id}";
        } elseif ($source_type === 'upload') {
            $file_name = basename($files['name'][$i]);
            $file_ext = pathinfo($file_name, PATHINFO_EXTENSION);
            $file_path = $upload_dir . $project_id . "-" . $entry_id . "." . $file_ext;
            $source_url = $file_path;
            $original_name = pathinfo($file_name, PATHINFO_FILENAME);

            move_uploaded_file($files['tmp_name'][$i], $file_path);
        }

        $stmt = $pdo->prepare("INSERT INTO project_entries (
            project_id, entry_id, source_type, source_url, original_name
        ) VALUES (?, ?, ?, ?, ?)");

        $stmt->execute([$project_id, $entry_id, $source_type, $source_url, $original_name]);
    }

    header("location: projects.php");
    exit;
}

include("header.php");
?>

<!-- Your existing HTML form can go here if needed -->

<?php
$scripts = ['create_project.js'];
include("scripts.php");
$tooltip = '';
include("footer.php");
ob_end_flush(); // End output buffering after all headers
?>

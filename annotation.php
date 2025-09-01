<?php
    $title = 'Platform for Affective Game ANnotation';
    $css = ['annotation.css'];
    $test_mode = !empty($_GET['test_mode']) ? 1 : 0;

    include("base.php");

    if (!isset($_COOKIE['seen_notice'])) {
        setcookie('seen_notice', 'seen', strtotime('+365 days'), '/', $_SERVER['HTTP_HOST']);
    }

    require_once "config.php";

    $session_id = $_COOKIE['session_id'] ?? getGUID();
    $progress = [];
    $started_project = 0;

    // Init globals
    $project_name = $project_id = $entry_id = $target = $type = $source_type = $video_loading = "";
    $n_of_entries = $n_of_participant_runs = $source = $sequence_n = $endless = $sound = $message = "";

    if ($_SERVER["REQUEST_METHOD"] === "GET") {
        $project_id = htmlspecialchars($_GET['id'], ENT_QUOTES, "UTF-8");
        $progress_entry = new stdClass();

        if ($test_mode > 0) {
            $project_name = htmlspecialchars($_GET['project_name'], ENT_QUOTES, "UTF-8");
            $target = htmlspecialchars($_GET['target'], ENT_QUOTES, "UTF-8");
            $type = htmlspecialchars($_GET['type'], ENT_QUOTES, "UTF-8");
            $sound = htmlspecialchars($_GET['sound'], ENT_QUOTES, "UTF-8");
            $source_type = htmlspecialchars($_GET['source_type'], ENT_QUOTES, "UTF-8");
            $video_loading = htmlspecialchars($_GET['video_loading'], ENT_QUOTES, "UTF-8");
            $source_url = htmlspecialchars($_GET['source'], ENT_QUOTES, "UTF-8");
            $source = ['source_url' => "https://www.youtube.com/watch?v=" . $source_url];
            $source_type = "youtube";
            $video_loading = "random";
            $endless = "on";
            $n_of_entries = 1;
            $n_of_participant_runs = 1;
        } else {
            $entry_id = $_GET['entry'] ?? '';
            $entry_id = htmlspecialchars($entry_id, ENT_QUOTES, "UTF-8");

            $stmt = $pdo->prepare("SELECT * FROM projects WHERE project_id = :project_id LIMIT 1");
            $stmt->bindParam(":project_id", $project_id, PDO::PARAM_STR);
            $stmt->execute();

            if ($stmt->rowCount() === 1) {
                $row = $stmt->fetch();
                $project_name = $row['project_name'];
                $target = $row['target'];
                $type = $row['type'];
                $source_type = $row['source_type'];
                $video_loading = $row['video_loading'];
                $endless = $row['endless'];
                $n_of_entries = $row['n_of_entries'];
                $n_of_participant_runs = $row['n_of_participant_runs'];
                $sound = $row['sound'];
                $message = $row['start_message'];
            } else {
                echo "Oops! Something went wrong. Please try again later.";
            }
            unset($stmt);

            // Handle progress cookie
            if (isset($_COOKIE['progress'])) {
                $progress = json_decode($_COOKIE['progress'], true);
                $this_project = "";

                foreach ($progress as $key => $entry) {
                    if ($entry['project_id'] === $project_id) {
                        $started_project = 1;
                        $this_project = $project_id;
                        $current_run = $entry;

                        if ($current_run['n_runs'] >= $n_of_participant_runs && $endless !== 'on') {
                            header("location: end.php?id=" . $project_id);
                            exit();
                        } elseif ($current_run['n_runs'] >= $n_of_participant_runs && $endless === 'on') {
                            $current_run['n_runs'] = 0;
                            $current_run['seen'] = [];
                            $progress[$key] = $current_run;
                            setcookie('progress', json_encode($progress), strtotime('+7 days'), '/', $_SERVER['HTTP_HOST']);
                        }
                        break;
                    }
                }

                if (strlen($this_project) < 1) {
                    $progress_entry->project_id = $project_id;
                    $progress_entry->seen = [];
                    $progress_entry->n_runs = 0;
                    $progress[] = $progress_entry;
                    setcookie('progress', json_encode($progress), strtotime('+7 days'), '/', $_SERVER['HTTP_HOST']);
                }
            } else {
                $progress_entry->project_id = $project_id;
                $progress_entry->seen = [];
                $progress_entry->n_runs = 0;
                $progress[] = $progress_entry;
                setcookie('progress', json_encode($progress), strtotime('+7 days'), '/', $_SERVER['HTTP_HOST']);

                $current_run = [
                    'project_id' => $project_id,
                    'n_runs' => 0,
                    'seen' => []
                ];
            }

            // Get available entries
            $available_entries = [];
            $stmt = $pdo->prepare("SELECT * FROM project_entries WHERE project_id = :project_id");
            $stmt->bindParam(":project_id", $project_id, PDO::PARAM_STR);
            if ($stmt->execute()) {
                while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
                    if (!in_array($row['entry_id'], $current_run['seen'])) {
                        $available_entries[] = $row;
                    }
                }
            } else {
                echo "Oops! Something went wrong. Please try again later.";
            }
            unset($stmt);

            // Select entry
            if (empty($entry_id)) {
                if ($source_type === "user_upload" || $source_type === "user_youtube") {
                    header("location: upload.php?id=" . $project_id);
                    exit();
                } else {
                    $source = ($video_loading === "sequence") ? reset($available_entries) : $available_entries[array_rand($available_entries)];
                }
            } else {
                $stmt = $pdo->prepare("SELECT * FROM project_entries WHERE project_id = :project_id AND entry_id = :entry_id LIMIT 1");
                $stmt->bindParam(":project_id", $project_id, PDO::PARAM_STR);
                $stmt->bindParam(":entry_id", $entry_id, PDO::PARAM_STR);
                $stmt->execute();
                if ($stmt->rowCount() === 1) {
                    $source = $stmt->fetch();
                } else {
                    echo "Oops! Something went wrong. Please try again later.";
                }
                unset($stmt);
            }

            unset($pdo);
        }
    }

    include("header.php");
?>

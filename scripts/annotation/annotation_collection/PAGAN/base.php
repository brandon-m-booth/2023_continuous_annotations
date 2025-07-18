<?php
# Generates GUID for username
function getGUID(){
    mt_srand((int)(microtime(true)*10000));//optional for php 4.2.0 and up.
    $charid = strtoupper(md5(uniqid(rand(), true)));
    $hyphen = chr(45);
    $uuid = substr($charid, 0, 8).$hyphen
        	.substr($charid, 8, 4).$hyphen
        	.substr($charid,12, 4).$hyphen
                .substr($charid,16, 4).$hyphen
        	.substr($charid,20,12);
    return $uuid;
}

function endsWith($str, $search) {
    return substr_compare($str, $search, -strlen($search)) === 0;
}
?>

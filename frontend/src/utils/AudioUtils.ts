function padTime(time: number) {
    return String(time).padStart(2, "0");
}

export function formatAudioTimestamp(time: number) {
    const date = new Date();
    date.setHours(0, 0, 0, 0);
    date.setSeconds(time);
    
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const seconds = date.getSeconds();
    
    const formattedTime = `${padTime(hours)}:${padTime(minutes)}:${padTime(seconds)}`;
    return formattedTime;
}

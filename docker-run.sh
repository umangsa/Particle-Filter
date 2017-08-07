if [ "$#" -eq 1 ] && [ $1 == "build" ]; then
    docker build --build-arg http_proxy --build-arg https_proxy --rm -t carnd-ekf .
fi
docker rm -f carnd-ekf
docker run --rm --name carnd-ekf -v `pwd`:/project -it -e http_proxy -e https_proxy -p 4567:4567 carnd-ekf

from padma.app import app


# start the web server
if __name__ == '__main__':
    print('* Starting web service...')
    app.run(port=5002, debug=True)

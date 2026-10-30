pipeline {
    agent any

    environment {
        IMAGE = "aikoru-ci"
        // Для push: в Jenkins создать credentials типа Username/Password с id dockerhub
        DOCKERHUB_REPO = "damirasainov/aikoru-ci"
    }

    triggers { pollSCM('H/2 * * * *') }

    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Build image') {
            steps { sh 'docker build -t $IMAGE:$BUILD_NUMBER -t $IMAGE:latest .' }
        }
        stage('Lint') {
            steps { sh 'docker run --rm $IMAGE:$BUILD_NUMBER ruff check aikoru tests' }
        }
        stage('Unit tests') {
            steps { sh 'docker run --rm $IMAGE:$BUILD_NUMBER pytest tests -v' }
        }
        stage('Push to Docker Hub') {
            when { anyOf { branch 'main'; expression { env.GIT_BRANCH?.endsWith('main') } } }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub',
                        usernameVariable: 'DH_USER', passwordVariable: 'DH_PASS')]) {
                    sh '''
                        echo "$DH_PASS" | docker login -u "$DH_USER" --password-stdin
                        docker tag $IMAGE:$BUILD_NUMBER $DOCKERHUB_REPO:$BUILD_NUMBER
                        docker tag $IMAGE:$BUILD_NUMBER $DOCKERHUB_REPO:latest
                        docker push $DOCKERHUB_REPO:$BUILD_NUMBER
                        docker push $DOCKERHUB_REPO:latest
                    '''
                }
            }
        }
    }
    post {
        always { sh 'docker image prune -f || true' }
    }
}

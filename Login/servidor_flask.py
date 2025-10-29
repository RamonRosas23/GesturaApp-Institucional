import hashlib
import hmac
import os
import base64
import token
from urllib.parse import quote, quote_plus
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, url_for
from itsdangerous import URLSafeTimedSerializer
import requests
from requests_oauthlib import OAuth2Session
import mysql.connector
from flask_mail import Mail, Message

app = Flask(__name__)
# Cargar variables de entorno
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)
app.secret_key = os.getenv('SECRET_KEY')

# Conectar con MySQL
def connect_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', '127.0.0.1'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''), 
        database=os.getenv('DB_NAME'),
        port=os.getenv('DB_PORT', '3306'),
        use_pure=True
    )

# Ruta de callback para Google
@app.route('/callback/google')
def google_callback():
    client_id = os.getenv('GOOGLE_CLIENT_ID')
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET')

    if not client_id or not client_secret:
        return "Error: GOOGLE_CLIENT_ID o GOOGLE_CLIENT_SECRET no están configurados correctamente.", 500

    redirect_uri = 'http://localhost:8000/callback/google'
    token_url = 'https://oauth2.googleapis.com/token'

    # Obtener el código y el estado de la URL
    code = request.args.get('code')
    error = request.args.get('error')  # Aquí se captura cualquier error de Google

    if error:
        # Si Google devolvió un error (ej. el usuario denegó permisos)
        return redirect("http://localhost:8080/google_error")  # Redirigir a PyQt con un mensaje de error

    state = request.args.get('state')

    if not state:
        return "Error: Estado de la autenticación no encontrado", 400

    # Crear la sesión OAuth2 con el estado recuperado
    scope = [
        "openid",
        "https://www.googleapis.com/auth/user.gender.read",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/user.phonenumbers.read",
        "https://www.googleapis.com/auth/user.birthday.read",
        "https://www.googleapis.com/auth/userinfo.profile"
    ]

    google = OAuth2Session(client_id, redirect_uri=redirect_uri, state=state, scope=scope)

    try:
        token = google.fetch_token(
            token_url,
            client_secret=client_secret,
            code=code,
            include_granted_scopes=True
        )
    except Exception as e:
        # En caso de error durante la obtención del token, redirigir a la pantalla de error
        return redirect("http://localhost:8080/google_error")

    # Obtener el perfil del usuario
    user_info = google.get('https://www.googleapis.com/oauth2/v1/userinfo').json()

    # Obtener más información del perfil del usuario
    extra_info = google.get('https://people.googleapis.com/v1/people/me?personFields=birthdays,genders,phoneNumbers').json()
    
    # Extraer la información adicional
    fecha_nacimiento = extra_info.get('birthdays', [{}])[0].get('date', None)
    genero = extra_info.get('genders', [{}])[0].get('formattedValue', '') 
    telefono = extra_info.get('phoneNumbers', [{}])[0].get('value', 'Sin teléfono')  # Valor predeterminado
    avatar = user_info.get('picture', None)  # Campo de avatar
    email = user_info['email']
    print(email)
    print(avatar)
    
    # Convertir la fecha de nacimiento en un formato adecuado para la base de datos
    if fecha_nacimiento:
        fecha_nacimiento = f"{fecha_nacimiento['year']}-{fecha_nacimiento['month']:02d}-{fecha_nacimiento['day']:02d}"
        
    # Conectar a la base de datos
    conn = connect_db()
    cursor = conn.cursor()

    # Verificar si el usuario ya está registrado
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user_data = cursor.fetchone()
    
    # Extraer los valores de user_id
    if user_data:
        user_id = user_data[0]  # Asumiendo que user_id es la primera columna
    else:
        user_id = None

    # Usuario ya existe, obtener datos sin convertir avatar a Base64 si es una URL
    avatar_base64 = None

    # Descargar la imagen desde la URL
    response = requests.get(avatar)

    if response.status_code == 200:
        # Convertir los datos binarios de la imagen a base64
        avatar_base64 = base64.b64encode(response.content).decode('utf-8')
    else:
        print("No se pudo descargar la imagen")
    
    if user_data:
        # Actualizar datos con información de Google si es necesario
        cursor.execute("""
            UPDATE users SET avatar=%s WHERE email=%s
        """, (avatar_base64, email))
        conn.commit()
    else:
        # Si el usuario no existe, registrarlo
        cursor.execute("""
            INSERT INTO users (username, email, nombres, apellidos, avatar, genero, fecha_nacimiento, telefono, direccion, tipo_registro)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_info['name'], user_info['email'], user_info.get('given_name', ''), user_info.get('family_name', ''), 
              user_info.get('picture', None), genero, fecha_nacimiento, telefono, 'Sin dirección', 'google'))
        conn.commit()
    conn.close()
    
    user_profile = {
            'user_id': str(user_id),
            'username': user_info['name'],
            'email': user_info['email'],
            'nombres': user_info.get('given_name', ''),
            'apellidos': user_info.get('family_name', ''),
            'avatar': avatar_base64,
            'telefono': telefono,
            'direccion': 'Sin dirección',
            'genero': genero,
            'fecha_nacimiento': fecha_nacimiento
        }

    # Redirigir a la aplicación PyQt con los datos del perfil del usuario
    redirect_url = f"http://localhost:8080/google_success?user_id={quote(user_profile['user_id'])}&username={quote(user_profile['username'])}&email={quote(user_profile['email'])}&avatar={quote(user_profile['avatar'])}&first_name={quote(user_profile['nombres'])}&last_name={quote(user_profile['apellidos'])}&gender={quote(user_profile['genero'])}&birthday={quote(user_profile['fecha_nacimiento'])}&direccion={quote(user_profile['direccion'])}"
    return redirect(redirect_url)


# Ruta de callback para Facebook
@app.route('/callback/facebook')
def facebook_callback():
    client_id = os.getenv('FACEBOOK_APP_ID')
    client_secret = os.getenv('FACEBOOK_APP_SECRET')

    if not client_id or not client_secret:
        return "Error: FACEBOOK_APP_ID o FACEBOOK_APP_SECRET no están configurados correctamente.", 500

    redirect_uri = 'http://localhost:8000/callback/facebook'
    token_url = 'https://graph.facebook.com/v12.0/oauth/access_token'

    # Obtener el código y el estado de la URL
    code = request.args.get('code')
    error = request.args.get('error')

    if error:
        return redirect("http://localhost:8080/facebook_error")

    state = request.args.get('state')

    if not state:
        return "Error: Estado de la autenticación no encontrado", 400

    # Crear la sesión OAuth2 con el estado recuperado
    scope = [
        'email', 
        'public_profile', 
        'user_birthday', 
        'user_gender', 
        'user_photos', 
        'user_hometown', 
        'user_location'
    ]

    facebook = OAuth2Session(client_id, redirect_uri=redirect_uri, state=state, scope=scope)

    try:
        # Obtener el token de acceso
        token = facebook.fetch_token(
            token_url,
            client_secret=client_secret,
            code=code
        )
        access_token = token['access_token']
    except Exception as e:
        return redirect("http://localhost:8080/facebook_error")

    # Calcular appsecret_proof
    appsecret_proof = hmac.new(
        client_secret.encode('utf-8'),
        msg=access_token.encode('utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()

    # Obtener el perfil del usuario con campos adicionales
    try:
        # Obtener el perfil del usuario con campos adicionales, incluyendo un avatar de mayor resolución
        user_info = facebook.get(f'https://graph.facebook.com/me?fields=id,name,email,picture.width(400).height(400),first_name,last_name,gender,birthday,hometown,location&appsecret_proof={appsecret_proof}').json()
    except Exception as e:
        return "Error al obtener el perfil de usuario de Facebook", 500

    # Extraer información
    email = user_info.get('email', None)
    first_name = user_info.get('first_name', None)
    last_name = user_info.get('last_name', None)
    gender = user_info.get('gender', None)
    birthday = user_info.get('birthday', None)
    hometown = user_info.get('hometown', {}).get('name', None)
    location = user_info.get('location', {}).get('name', None)
    
    # Extraer información adicional sobre si el avatar es una silueta
    avatar_data = user_info.get('picture', {}).get('data', {})
    is_silhouette = avatar_data.get('is_silhouette', False)

    # Verificar si es la silueta predeterminada
    if is_silhouette:
        avatar_url = None
    else:
        # Utiliza el URL del avatar
        avatar_url = avatar_data.get('url')

    if not email:
        return "Error: No se pudo obtener el correo electrónico de Facebook", 500

    # Conectar a la base de datos y verificar/registrar al usuario
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
    user_data = cursor.fetchone()
    
    # Extraer los valores de user_id
    if user_data:
        user_id = user_data[0]  # Asumiendo que user_id es la primera columna
    else:
        user_id = None

    if user_data:
        cursor.execute("""
            UPDATE users SET avatar=%s WHERE email=%s
        """, (avatar_url, email))
        conn.commit()
    else:
        cursor.execute("""
            INSERT INTO users (username, email, avatar, tipo_registro, nombres, apellidos, genero, fecha_nacimiento, direccion)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (f'{first_name} {last_name}', email, avatar_url, 'facebook', first_name, last_name, gender, birthday, location))
        conn.commit()

        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user_data = cursor.fetchone()

    # Definir el perfil de usuario independientemente de si es un usuario nuevo o existente
    user_profile = {
        'user_id': str(user_id),
        'username': f'{first_name} {last_name}',
        'email': email,
        'nombres': first_name,
        'apellidos': last_name,
        'avatar': avatar_url,  # Avatar convertido a base64
        'telefono': 'No disponible',
        'direccion': hometown or location or 'No disponible',
        'genero': gender or 'No disponible',
        'fecha_nacimiento': birthday or 'No disponible'
    }

    conn.close()


    # Codificar avatar_url antes de crear el redirect_url
    avatar_url_encoded = quote_plus(user_profile['avatar'])

    # Crear el redirect URL codificando el avatar URL
    redirect_url = (
        f"http://localhost:8080/facebook_success?"
        f"user_id={quote_plus(user_profile['user_id'])}&"
        f"username={quote_plus(user_profile['username'])}&"
        f"email={quote_plus(user_profile['email'])}&"
        f"avatar={avatar_url_encoded}&"
        f"first_name={quote_plus(user_profile['nombres'])}&"
        f"last_name={quote_plus(user_profile['apellidos'])}&"
        f"gender={quote_plus(user_profile['genero'])}&"
        f"birthday={quote_plus(user_profile['fecha_nacimiento'])}&"
        f"direccion={quote_plus(user_profile['direccion'])}"
    )

    return redirect(redirect_url)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = os.getenv('EMAIL_USER')
app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')

mail = Mail(app)

# Ruta para servir archivos desde la carpeta assets
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(os.path.join(app.root_path, 'assets'), filename)

def send_reset_email_to_user(user_email, reset_url, user_name):
    msg = Message('Recuperación de contraseña', sender=os.getenv('EMAIL_USER'), recipients=[user_email])
    
    # Enlace directo de la imagen en Imgur
    logo_url = "https://i.imgur.com/it44PK3.png"  # Cambia "xxxxx.png" por el ID de la imagen en Imgur

    # Cuerpo del mensaje en HTML con la imagen embebida desde Imgur
    msg.html = f'''
    <html>
    <body>
        <div style="font-family: Arial, sans-serif; text-align: center; background-color: #ffffff; padding: 20px;">
            <img src="{logo_url}" alt="Logo" style="width: 150px; height: auto;"/>
            <h2 style="color: #333;">Hola {user_name}</h2>
            <p style="color: #333; font-size: 16px;">
                Hemos recibido una solicitud para restablecer la contraseña de tu cuenta.
            </p>
            <p style="color: #333; font-size: 16px;">
                Por favor, utiliza el siguiente botón para iniciar el proceso de recuperación de contraseña:
            </p>
            <a href="{reset_url}" style="background-color: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-size: 18px;">
                Restablecer contraseña
            </a>
            <p style="color: #999; font-size: 12px; margin-top: 20px;">
                Si no solicitaste este cambio, ignora este correo.
            </p>
        </div>
    </body>
    </html>
    '''
    
    mail.send(msg)

def verify_reset_token(token, expiration=3600):
    s = URLSafeTimedSerializer(app.secret_key)
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=expiration)
        return email
    except Exception as e:
        print(f"Error verificando el token: {e}")
        return None


def generate_reset_token(email):
    s = URLSafeTimedSerializer(app.secret_key)
    token = s.dumps(email, salt='password-reset-salt')
    return token


@app.route('/send_reset_email', methods=['POST'])
def send_reset_email():
    email = request.form['email']
    nombre = request.form.get('nombre', 'Usuario')  # Si no se envía el nombre, se usa 'Usuario' como predeterminado
    
    # Generar token para el email
    token = generate_reset_token(email)
    reset_url = f'http://localhost:8000/reset_password/{token}'
    
    # Enviar el correo con el nombre y el enlace de restablecimiento
    send_reset_email_to_user(email, reset_url, nombre)
    
    return token, 200

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    email = verify_reset_token(token)
    if not email:
        return "Token inválido o expirado", 400

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        conn = connect_db()
        cursor = conn.cursor()

        # Verificar la contraseña actual
        cursor.execute("SELECT password FROM users WHERE email=%s", (email,))
        stored_password = cursor.fetchone()[0]

        if stored_password != current_password:
            error_message = "La contraseña anterior no es correcta"
            return render_template('reset_password.html', error=error_message)

        if not new_password or not confirm_password:
            error_message = "Faltan campos de contraseña"
            return render_template('reset_password.html', error=error_message)

        if new_password != confirm_password:
            error_message = "Las contraseñas no coinciden"
            return render_template('reset_password.html', error=error_message)

        # Actualizar la contraseña en la base de datos
        cursor.execute("UPDATE users SET password=%s WHERE email=%s", (new_password, email))
        conn.commit()
        conn.close()

        return "Contraseña actualizada exitosamente", 200

    return render_template('reset_password.html')

if __name__ == '__main__':
    app.run(port=8000, debug=True)

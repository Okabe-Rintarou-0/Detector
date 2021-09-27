package com.example.detector.activity;

import android.Manifest;
import android.content.ContentUris;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.detector.R;
import com.example.detector.utils.AssetsUtil;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class DetectActivity extends AppCompatActivity {
    public static final int CHOOSE_PHOTO = 2;
    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private ImageView orgImg;
    private ImageView outImg;
    private String imagePath = null;
    private static final String[] PERMISSIONS_STORAGE = {Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE};
    private Module module;
    private Bitmap curBitmap;
    private static final float THRESH = 1.5f;

    private void loadModule() {
        String moduleAbsoluteFilePath = AssetsUtil.assetFilePath(this, "model.ptl");
        moduleAbsoluteFilePath = moduleAbsoluteFilePath == null ? "null" : moduleAbsoluteFilePath;
        Log.d("path", moduleAbsoluteFilePath);
        module = Module.load(moduleAbsoluteFilePath);
    }

    private int transform(float v) {
        float sig_v = sigmoid(v);
        float r = sig_v * 0.33f;
        float g = sig_v * 0.33f;
        float b = sig_v * 0.33f;
        return Color.rgb(r, g, b);
    }

    public static float sigmoid(float value) {
        //Math.E=e;Math.Pow(a,b)=a^b
        float ey = (float) Math.pow(Math.E, -value);
        return 1 / (1 + ey);
    }

    private Bitmap floatArrayToBitmap(float[] floatArray, int width, int height) {
        // You are using RGBA that's why Config is ARGB.8888
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        // vector is your int[] of ARGB
        List<Integer> intList = new ArrayList<>();
        for (float v : floatArray) intList.add(transform(v));
        int[] intArray = intList.stream().mapToInt(Integer::intValue).toArray();
        System.out.println(Arrays.toString(intArray));
        bitmap.copyPixelsFromBuffer(IntBuffer.wrap(intArray));
        return bitmap;
    }


    private void detect(Bitmap bitmap) {
        System.out.println(bitmap.getPixel(0, 0));
        bitmap = Bitmap.createScaledBitmap(bitmap, 608, 608, false);
        outImg.setImageBitmap(bitmap);
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        System.out.println(Arrays.toString(inputTensor.shape()));
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
        System.out.println(Arrays.toString(outputTensor.shape()));
        float[] floatArray = outputTensor.getDataAsFloatArray();
        System.out.println(Arrays.toString(floatArray));
        bitmap = floatArrayToBitmap(floatArray, 608, 608);
        outImg.setImageBitmap(bitmap);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_album);

        Button choosePhotoButton = findViewById(R.id.choose_photo);
        Button detectButton = findViewById(R.id.detect_button);
        orgImg = findViewById(R.id.org_img);
        outImg = findViewById(R.id.out_img);
        detectButton.setOnClickListener(v -> detect(curBitmap));
        choosePhotoButton.setOnClickListener(v -> tryOpenAlbum());
        orgImg.setOnClickListener(v -> tryOpenAlbum());

        loadModule();
    }

    private void tryOpenAlbum() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, PERMISSIONS_STORAGE, REQUEST_EXTERNAL_STORAGE);
        } else {
            openAlbum();
        }
    }

    private void openAlbum() {
        Intent intent = new Intent("android.intent.action.GET_CONTENT");
        intent.setType("image/*");
        startActivityForResult(intent, CHOOSE_PHOTO); //打开相册
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 1) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openAlbum();
            } else {
                Toast.makeText(this, "你拒绝了该权限", Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CHOOSE_PHOTO) {
            if (resultCode == RESULT_OK) {
                //判断手机系统版本号
                if (data != null) {
                    //4.4及以上系统使用这个方法处理图片
                    handleImageOnKitKat(data);
                }
            }
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    private void handleImageOnKitKat(Intent data) {
        Uri uri = data.getData();
        if (DocumentsContract.isDocumentUri(this, uri)) {
            //如果是document类型的Uri，则通过document id处理
            String docId = DocumentsContract.getDocumentId(uri);
            if ("com.android.providers.media.documents".equals(uri.getAuthority())) {
                String id = docId.split(":")[1];  //解析出数字格式的id
                String selection = MediaStore.Images.Media._ID + "=" + id;
                imagePath = getImagePath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, selection);
            } else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())) {
                Uri contentUri = ContentUris.withAppendedId(Uri.parse("content://downloads/public downloads"), Long.parseLong(docId));
                imagePath = getImagePath(contentUri, null);
            }
        } else if ("content".equalsIgnoreCase(uri.getScheme())) {
            //如果是file类型的Uri，直接获取图片路径即可
            imagePath = getImagePath(uri, null);
        } else if ("file".equalsIgnoreCase(uri.getScheme())) {
            //如果是file类型的Uri，直接获取图片路径即可
            imagePath = uri.getPath();
        }
        displayImage(imagePath); //根据图片路径显示图片
    }

    //将选择的图片Uri转换为路径
    private String getImagePath(Uri uri, String selection) {
        String path = null;
        //通过Uri和selection来获取真实的图片路径
        Cursor cursor = getContentResolver().query(uri, null, selection, null, null);
        if (cursor != null) {
            if (cursor.moveToFirst()) {
                path = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
            }
            cursor.close();
        }
        return path;
    }

    //展示图片
    private void displayImage(String imagePath) {
        if (imagePath != null && !imagePath.equals("")) {
            curBitmap = BitmapFactory.decodeFile(imagePath);
            orgImg.setImageBitmap(curBitmap);
            //存储上次选择的图片路径，用以再次打开app设置图片
            SharedPreferences sp = getSharedPreferences("sp_img", MODE_PRIVATE);  //创建xml文件存储数据，name:创建的xml文件名
            SharedPreferences.Editor editor = sp.edit(); //获取edit()
            editor.putString("imgPath", imagePath);
            editor.apply();
        } else {
            Toast.makeText(this, "获取图片失败", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    //当该activity与用户能进行交互时被执行，用户可以获得activity的焦点，能够与用户交互。
    protected void onResume() {
        super.onResume();
        //设置再次app时显示的图片
        SharedPreferences sp = getSharedPreferences("sp_img", MODE_PRIVATE);
        //取出上次存储的图片路径设置此次的图片展示
        String lastImagePath = sp.getString("imgPath", null);
        displayImage(lastImagePath);
    }
}